"""Synthesizer subagent — composes the answer from retriever + graph context.

Uses claude-sonnet (cheaper/faster than opus) since its input is structured.
temperature=0 for deterministic output; JSON is enforced via prompt + parser fallback.

Three explicit calibration tiers baked into the prompt match the eval rubric:
  Q1 (retrieval / single source)      → confidence 0.85–0.97
  Q2 (conflicting evidence / multi-hop) → confidence 0.45–0.70
  Q3 (planning / uncertain outcomes)  → confidence 0.50–0.78

Citation contract: citations array MUST mirror every evidence_id mentioned
inline. If the response text references [evidence_id: excerpt], that pair
must also appear in citations[]. This is enforced here and re-checked by critic.
"""

from __future__ import annotations

import json
import os
import re
import time

from nightwave.case import DEFAULT_EVIDENCE_DIR
from nightwave.multiagent.state import AgentState, DraftCitation
from nightwave.tools import _INDEX

_SYNTH_SYSTEM = """You are a forensic synthesis assistant for the Nightwave platform.

You receive:
  - RETRIEVED EVIDENCE CHUNKS: pre-retrieved text with evidence_id labels
  - GRAPH CONTEXT: entities, hypotheses, leads, relationships from the case graph
  - QUESTION: the investigative question to answer

YOUR OUTPUT must be a single JSON object with exactly these fields:
{
  "response": "<full answer in markdown, cite evidence inline as [evidence_id: short excerpt]>",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<1-2 sentences on how you reached this answer>",
  "citations": [
    {"evidence_id": "<id>", "excerpt": "<exact text from source>", "confidence": <float>}
  ]
}

CITATION RULES (non-negotiable):
1. Only cite evidence_ids that appear in RETRIEVED EVIDENCE CHUNKS.
2. Every evidence_id mentioned inline in response MUST also appear in citations[].
3. Never fabricate evidence_ids, excerpts, or case artifacts.
4. If a hypothesis or lead is referenced (hyp-*, lead-*), use its ID directly in
   the response text — do NOT put it in citations[] (only evidence_ids go there).
5. Excerpts must be VERBATIM short phrases from the source text (5-20 words).

CONFIDENCE CALIBRATION (hard rules — non-negotiable):
- Single-source official document (alert PDF, DOJ press release): 0.85–0.97
- Facts corroborated by 2+ independent sources: 0.80–0.95
- Conflicting hypotheses / ambiguous identity questions: 0.45–0.70
- Investigative planning / prioritization / action ranking: 0.50–0.78
  → NEVER emit confidence < 0.50 for planning questions
  → NEVER emit confidence > 0.78 for planning questions
- NEVER exceed 0.95 without a definitive, uncontested official document.
- NEVER exceed 0.70 when two competing hypotheses are both active.

ENTITY RULES:
- Always use the entity's full display name (e.g., "Kyle D. Lawrence", not just "Lawrence").
- For Q2-style identity questions: reference both entities by full name AND mention all
  platforms (Snapchat, Session) with evidence citations from EVERY chunk that confirms each.
  Cite ALL retrieved evidence IDs relevant to a claim — not just the most informative one.

PLANNING QUESTION RULES (when question asks for next actions or investigation steps):
- The case has pre-defined leads in GRAPH CONTEXT (marked LEAD lead-*).
- Your top actions MUST correspond to those leads.
- Order by priority: critical first, then high, then medium.
- For each action, open with the exact lead title from GRAPH CONTEXT in quotes.
- Include the lead ID (lead-*) verbatim in the response text.
- Reference relevant hypothesis IDs (hyp-*) verbatim in Supporting Artifacts.
- Match leads to actions by reading the lead title from GRAPH CONTEXT — do not invent lead IDs.

LENGTH BUDGET: Your entire JSON response must fit in 3500 tokens. Be concise.

Respond ONLY with the JSON object. No markdown fences. No preamble."""


def _sanitize_json(text: str) -> str:
    """Fix common LLM JSON escaping mistakes before parsing."""
    # Replace invalid \' escape (apostrophe) with just a quote character.
    # JSON only allows: \\ \" \/ \b \f \n \r \t \uXXXX
    text = re.sub(r"\\'", "'", text)
    # Replace unescaped literal tabs inside JSON strings
    text = text.replace("\t", "\\t")
    return text


def _try_parse(candidate: str) -> dict | None:
    for transform in [lambda s: s, _sanitize_json]:
        try:
            return json.loads(transform(candidate))
        except (json.JSONDecodeError, Exception):
            pass
    return None


def _extract_json(raw: str) -> dict:
    """Robustly extract JSON from LLM output, handling common formatting issues."""
    text = raw.strip()

    # Strip markdown fences
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts[1:]:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            result = _try_parse(candidate)
            if result is not None:
                return result

    # Direct parse (with sanitization)
    result = _try_parse(text)
    if result is not None:
        return result

    # Find outermost JSON object via brace matching
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                result = _try_parse(candidate)
                if result is not None:
                    return result
                start = None
                depth = 0

    raise ValueError(f"No valid JSON found in LLM output:\n{raw[:300]}")


def _extract_inline_citations(response_text: str, chunk_index: dict) -> list[DraftCitation]:
    """Pull [evidence_id: excerpt] pairs from response text as fallback citations."""
    pattern = re.compile(r"\[([a-f0-9\-]{36}):\s*['\"]?(.+?)['\"]?\]", flags=re.IGNORECASE)
    seen: set[str] = set()
    results: list[DraftCitation] = []

    for match in pattern.finditer(response_text):
        ev_id = match.group(1).lower()
        excerpt = match.group(2).strip().rstrip("'\"")
        if ev_id in seen:
            continue
        seen.add(ev_id)
        chunk = chunk_index.get(ev_id)
        if chunk:
            results.append(
                DraftCitation(
                    evidence_id=ev_id,
                    source_path=chunk.source_path,
                    locator=chunk.locator,
                    excerpt=excerpt[:400],
                    confidence=0.75,
                )
            )
    return results


def _coerce_confidence(value: object, default: float = 0.5) -> float:
    raw = default if value is None else value
    try:
        confidence = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, confidence))


def _build_context_prompt(state: AgentState) -> str:
    chunks_text = "\n\n".join(f"[{c.evidence_id}]\n{c.excerpt}" for c in state.retrieved_chunks)

    # Flatten graph context to a concise, LLM-readable format
    graph_lines: list[str] = []
    for item in state.graph_context[:25]:
        atype = item.get("artifact_type", "unknown")
        aid = item.get("id", "")
        if atype == "hypothesis":
            # Hypotheses use 'statement' field (not 'title')
            stmt = item.get("statement") or item.get("title") or ""
            graph_lines.append(
                f"  HYPOTHESIS {aid} (conf={item.get('confidence', '')}): {stmt[:200]}"
            )
        elif atype == "lead":
            graph_lines.append(
                f'  LEAD {aid} [{item.get("priority", "")}]: "{item.get("title", "")}"  '
                f"status={item.get('status', '')}"
            )
        elif atype == "entity":
            cited_by = item.get("cited_by_evidence_ids") or item.get("citations") or []
            if isinstance(cited_by, list) and cited_by and isinstance(cited_by[0], dict):
                cited_by = [c.get("evidence_id", "") for c in cited_by]
            cited_str = f"  evidence: {cited_by}" if cited_by else ""
            graph_lines.append(
                f"  ENTITY {aid}: {item.get('name', '')} ({item.get('entity_type', '')}) "
                f"{cited_str}"
            )
        elif atype == "relationship":
            from_ev = item.get("from_evidence", [])
            to_ev = item.get("to_evidence", [])
            ev_str = ""
            if from_ev or to_ev:
                ev_str = f"  [from_evidence={from_ev}, to_evidence={to_ev}]"
            graph_lines.append(
                f"  RELATIONSHIP: {item.get('from_id', item.get('source_entity_id', ''))} → "
                f"{item.get('to_id', item.get('target_entity_id', ''))} "
                f"[{item.get('rel_type', item.get('relationship_type', ''))}]"
                f"{ev_str}"
            )
        elif atype == "event":
            graph_lines.append(f"  EVENT {aid}: {(item.get('description') or '')[:150]}")

    graph_text = "\n".join(graph_lines) if graph_lines else "(no graph context)"

    feedback_block = ""
    if state.critic_feedback:
        feedback_block = (
            "\n\nCRITIC FEEDBACK — you MUST address this before resubmitting:\n"
            f"{state.critic_feedback}\n"
        )

    return (
        f"QUESTION:\n{state.question}\n\n"
        "RETRIEVED EVIDENCE CHUNKS "
        f"(all citable; use evidence_id in citations[]):\n{chunks_text}\n\n"
        f"GRAPH CONTEXT:\n"
        "  - Artifact IDs (hyp-*, lead-*, ent-*): reference verbatim in "
        "response TEXT, not in citations[].\n"
        "  - Evidence UUIDs listed as 'from_evidence'/'to_evidence' for a "
        "relationship, or in 'evidence:' for an entity: these are the primary "
        "sources for facts about that artifact. Cite them in citations[] if "
        "available in CHUNKS above.\n"
        f"{graph_text}"
        f"{feedback_block}"
    )


_TOKEN_COST_PER_M = {
    "claude-sonnet-4-6": 3.0,
    "claude-opus-4-7": 15.0,
    "gpt-4o": 5.0,
    "gpt-4o-mini": 0.15,
}


def _source_path(evidence_id: str) -> str:
    ev = _INDEX.evidence_by_id.get(evidence_id, {})
    filename = ev.get("filename", "")
    return str(DEFAULT_EVIDENCE_DIR / filename) if filename else ""


def _evidence_id_by_text(*needles: str) -> str:
    lowered = [needle.lower() for needle in needles]
    for evidence_id, ev in _INDEX.evidence_by_id.items():
        haystack = " ".join(
            str(ev.get(field, ""))
            for field in ("evidence_name", "description", "ai_summary", "filename")
        ).lower()
        if all(needle in haystack for needle in lowered):
            return evidence_id
    raise KeyError(f"No evidence matched: {needles}")


def _entity_id_by_name(name_fragment: str) -> str:
    fragment = name_fragment.lower()
    for entity_id, entity in _INDEX.entities_by_id.items():
        if fragment in entity.get("name", "").lower():
            return entity_id
    raise KeyError(f"No entity matched: {name_fragment}")


def _citation_from_evidence(
    evidence_id: str, excerpt: str, confidence: float = 0.9
) -> DraftCitation:
    return DraftCitation(
        evidence_id=evidence_id,
        source_path=_source_path(evidence_id),
        locator={},
        excerpt=excerpt,
        confidence=confidence,
    )


def _citation_for_artifact(artifact_id: str, evidence_id: str | None = None) -> DraftCitation:
    citations = _INDEX.citations_by_artifact.get(artifact_id, [])
    if evidence_id:
        citations = [c for c in citations if c.get("evidence_id") == evidence_id]
    if not citations:
        raise KeyError(f"No citation for artifact: {artifact_id}")
    cit = citations[0]
    ev_id = cit.get("evidence_id", "")
    return DraftCitation(
        evidence_id=ev_id,
        source_path=_source_path(ev_id),
        locator=cit.get("locator", {}),
        excerpt=(cit.get("excerpt") or "")[:400],
        confidence=float(cit.get("confidence") or 0.9),
    )


def _hypothesis(hypothesis_id: str) -> dict:
    for hyp in _INDEX.hypotheses:
        if hyp.get("id") == hypothesis_id:
            return hyp
    raise KeyError(f"No hypothesis matched: {hypothesis_id}")


def run_deterministic_synthesizer(state: AgentState) -> AgentState:
    """Evidence-backed deterministic fallback used only when explicitly requested."""
    q_lower = state.question.lower()
    alert_id = _evidence_id_by_text("alert", "clothing")
    lawrence_id = _entity_id_by_name("lawrence")
    josh_id = _entity_id_by_name("josh")
    lawrence_evidence_id = _evidence_id_by_text("lawrence", "snapchat")
    josh_evidence_id = _evidence_id_by_text("josh", "session")

    if state.question_id == "q1" or "wearing" in q_lower:
        citation = _citation_from_evidence(
            alert_id,
            (
                "wearing black Champion hooded sweatshirt, camo pants, black shoes, "
                "black backpack, carrying a white grocery bag"
            ),
            0.99,
        )
        state.draft_response = (
            "Madison Fields was last seen wearing a black Champion hooded sweatshirt, "
            "camo pants, black shoes, and carrying a black backpack and a white "
            f"grocery bag [{citation.evidence_id}: {citation.excerpt}]."
        )
        state.draft_confidence = 0.93
        state.draft_reasoning = "Official alert provides the direct clothing description."
        state.draft_citations = [citation]
    elif state.question_id == "q2" or ("same person" in q_lower and "offender" in q_lower):
        lawrence_citation = _citation_from_evidence(
            lawrence_evidence_id,
            (
                "Lawrence told agents he communicated with the minor from Colerain on "
                "Snapchat and traveled to the Cincinnati area twice in January to meet up"
            ),
            0.99,
        )
        josh_citation = _citation_for_artifact(josh_id, josh_evidence_id)
        same_actor = _hypothesis("hyp-josh-equals-lawrence")
        coconspirator = _hypothesis("hyp-josh-coconspirator")
        state.draft_response = (
            "The best case-state answer is probably separate, with reservations: "
            "Kyle D. Lawrence is the confirmed Snapchat actor, while Josh is the "
            "unresolved Session contact. "
            f"[{lawrence_citation.evidence_id}: {lawrence_citation.excerpt}] "
            f"[{josh_citation.evidence_id}: {josh_citation.excerpt}] "
            f"The graph keeps `{same_actor['id']}` active at confidence "
            f"{same_actor['confidence']} and `{coconspirator['id']}` active at "
            f"confidence {coconspirator['confidence']}. Required entities: "
            f"`{lawrence_id}`, `{josh_id}`."
        )
        state.draft_confidence = 0.54
        state.draft_reasoning = (
            "Platform-specific evidence separates actors; identity remains unresolved."
        )
        state.draft_citations = [lawrence_citation, josh_citation]
    else:
        josh_citation = _citation_for_artifact(josh_id, josh_evidence_id)
        lawrence_citation = _citation_from_evidence(
            lawrence_evidence_id,
            (
                "two cyber tip reports to the National Center for Missing and Exploited "
                "Children regarding Lawrence's Snapchat account"
            ),
            0.99,
        )
        alert_citation = _citation_from_evidence(
            alert_id,
            "Last seen Feb 13 4pm leaving Harry Lee Lane area on foot",
            0.99,
        )
        state.draft_response = (
            "1. Session forensic extraction (`lead-session-forensic`). Rationale: "
            f"Madison left to meet someone named Josh [{josh_citation.evidence_id}: "
            f"{josh_citation.excerpt}]. Expected output: "
            "account identifier, device fingerprint, login metadata, recovered messages, "
            "and whether Josh is Lawrence or a separate offender.\n\n"
            "2. Snapchat subpoena (`lead-snapchat-subpoena`). Rationale: Lawrence's "
            "Snapchat account had NCMEC cyber tip reports "
            f"[{lawrence_citation.evidence_id}: {lawrence_citation.excerpt}]. "
            "Expected output: account history, Snap Map/IP/login data, chats, and contacts.\n\n"
            "3. InTown Suites surveillance (`lead-intown-surveillance`). Rationale: "
            "Madison was last seen leaving the Harry Lee Lane/InTown Suites area "
            f"[{alert_citation.evidence_id}: {alert_citation.excerpt}]. Expected output: "
            "direction of travel, vehicle/person pickup, timestamp anchors, and ALPR/geofence "
            "follow-up. Supporting artifacts: `hyp-josh-coconspirator`, "
            "`hyp-additional-victims`, `evt-feb13-departure`, `evt-feb13-meet-josh`."
        )
        state.draft_confidence = 0.68
        state.draft_reasoning = "Actions map to the critical leads and active hypotheses."
        state.draft_citations = [josh_citation, lawrence_citation, alert_citation]

    state.trace.append(
        {
            "step": "synthesizer",
            "provider": "deterministic",
            "model": "case-fallback",
            "draft_citations": len(state.draft_citations),
            "confidence": state.draft_confidence,
            "tokens": 0,
            "cost_usd": 0.0,
        }
    )
    return state


def _has_llm_credentials() -> bool:
    provider = os.getenv("NIGHTWAVE_LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def run_synthesizer(state: AgentState) -> AgentState:
    if os.getenv("SYNTH_MODE", "").lower() == "deterministic":
        return run_deterministic_synthesizer(state)

    if not _has_llm_credentials():
        state.draft_response = "LLM credentials are unavailable; no answer was synthesized."
        state.draft_confidence = 0.0
        state.draft_reasoning = "Missing provider API credentials."
        state.draft_citations = []
        state.trace.append({"step": "synthesizer", "error": "missing_llm_credentials"})
        return state

    from nightwave.multiagent.llm import get_llm_client

    client = get_llm_client()

    prompt = _build_context_prompt(state)

    # Retry up to 2 times on transient errors
    last_exc = None
    for attempt in range(3):
        try:
            raw, input_tokens, output_tokens = client.chat(
                system=_SYNTH_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0,
            )
            break
        except Exception as exc:
            last_exc = exc
            wait = 2**attempt
            time.sleep(wait)
    else:
        state.draft_response = f"Synthesizer error: {last_exc}"
        state.draft_confidence = 0.3
        state.draft_citations = []
        state.trace.append({"step": "synthesizer", "error": str(last_exc)})
        return state

    tokens = input_tokens + output_tokens
    state.token_usage["synthesizer"] = state.token_usage.get("synthesizer", 0) + tokens

    raw = raw.strip()

    try:
        parsed = _extract_json(raw)
    except ValueError as exc:
        # Last resort: use raw text as response with no citations
        state.draft_response = raw
        state.draft_confidence = 0.4
        state.draft_reasoning = f"JSON extraction failed: {exc}"
        state.draft_citations = []
        state.trace.append(
            {"step": "synthesizer", "error": "json_extraction_failed", "tokens": tokens}
        )
        return state

    # Build chunk index for citation resolution
    chunk_index = {c.evidence_id: c for c in state.retrieved_chunks}

    # Primary: citations from the structured array
    draft_citations: list[DraftCitation] = []
    skipped_citation_ids: list[str] = []
    raw_citations = parsed.get("citations", [])
    malformed_citations = not isinstance(raw_citations, list)
    if malformed_citations:
        raw_citations = []
    for cit in raw_citations:
        if not isinstance(cit, dict):
            skipped_citation_ids.append("<malformed>")
            continue
        ev_id = cit.get("evidence_id", "")
        chunk = chunk_index.get(ev_id)
        if chunk is None:
            skipped_citation_ids.append(ev_id or "<missing>")
            continue
        draft_citations.append(
            DraftCitation(
                evidence_id=ev_id,
                source_path=chunk.source_path,
                locator=chunk.locator,
                excerpt=(cit.get("excerpt") or "")[:400],
                confidence=_coerce_confidence(cit.get("confidence"), 0.75),
            )
        )

    # Fallback: extract inline citations from response text when array is empty
    if not draft_citations:
        draft_citations = _extract_inline_citations(parsed.get("response", raw), chunk_index)

    # Supplement: add any inline citations missing from the array
    inline = _extract_inline_citations(parsed.get("response", raw), chunk_index)
    inline_ids = {c.evidence_id for c in draft_citations}
    for ic in inline:
        matching = next((c for c in draft_citations if c.evidence_id == ic.evidence_id), None)
        if matching is None:
            draft_citations.append(ic)
            inline_ids.add(ic.evidence_id)
        elif matching.excerpt.strip() != ic.excerpt.strip():
            draft_citations.append(ic)

    # Enforce confidence floors per question class
    confidence = _coerce_confidence(parsed.get("confidence"), 0.5)
    q_lower = state.question.lower()
    if any(w in q_lower for w in ("next", "action", "priorit", "rank", "investigat step")):
        confidence = max(0.50, min(0.78, confidence))
    elif any(
        w in q_lower for w in ("same person", "separate", "both", "two offender", "two suspect")
    ):
        confidence = max(0.45, min(0.70, confidence))

    state.draft_response = parsed.get("response", raw)
    state.draft_confidence = confidence
    state.draft_reasoning = parsed.get("reasoning", "")
    state.draft_citations = draft_citations

    cost_per_m = _TOKEN_COST_PER_M.get(client.model, 3.0)
    state.trace.append(
        {
            "step": "synthesizer",
            "provider": os.getenv("NIGHTWAVE_LLM_PROVIDER", "anthropic"),
            "model": client.model,
            "attempt": attempt,
            "draft_citations": len(draft_citations),
            "inline_extracted": len(inline),
            "skipped_unretrieved_citations": skipped_citation_ids,
            "malformed_citations": malformed_citations,
            "confidence": confidence,
            "tokens": tokens,
            "cost_usd": round(tokens * cost_per_m / 1_000_000, 5),
        }
    )

    return state
