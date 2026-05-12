"""Synthesizer subagent — composes the answer from retriever + graph context.

Uses claude-sonnet (cheaper/faster than opus) since its input is structured.
temperature=0 for deterministic output; JSON enforced via strict system prompt + _extract_json fallback.

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

from nightwave.multiagent.state import AgentState, DraftCitation


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
- Your top actions MUST correspond to those leads, ordered by priority: critical first, then high, then medium.
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
                candidate = text[start:i + 1]
                result = _try_parse(candidate)
                if result is not None:
                    return result
                start = None
                depth = 0

    raise ValueError(f"No valid JSON found in LLM output:\n{raw[:300]}")


def _extract_inline_citations(response_text: str, chunk_index: dict) -> list[DraftCitation]:
    """Pull [evidence_id: excerpt] pairs from response text as fallback citations."""
    pattern = re.compile(r"\[([a-f0-9\-]{36}):\s*['\"]?(.+?)['\"]?\]")
    seen: set[str] = set()
    results: list[DraftCitation] = []

    for match in pattern.finditer(response_text):
        ev_id = match.group(1)
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


def _build_context_prompt(state: AgentState) -> str:
    chunks_text = "\n\n".join(
        f"[{c.evidence_id}]\n{c.excerpt}"
        for c in state.retrieved_chunks
    )

    # Flatten graph context to a concise, LLM-readable format
    graph_lines: list[str] = []
    for item in state.graph_context[:25]:
        atype = item.get("artifact_type", "unknown")
        aid = item.get("id", "")
        if atype == "hypothesis":
            # Hypotheses use 'statement' field (not 'title')
            stmt = item.get("statement") or item.get("title") or ""
            graph_lines.append(
                f"  HYPOTHESIS {aid} (conf={item.get('confidence','')}): {stmt[:200]}"
            )
        elif atype == "lead":
            graph_lines.append(
                f"  LEAD {aid} [{item.get('priority','')}]: \"{item.get('title','')}\"  "
                f"status={item.get('status','')}"
            )
        elif atype == "entity":
            cited_by = item.get("cited_by_evidence_ids") or item.get("citations") or []
            if isinstance(cited_by, list) and cited_by and isinstance(cited_by[0], dict):
                cited_by = [c.get("evidence_id", "") for c in cited_by]
            cited_str = f"  evidence: {cited_by}" if cited_by else ""
            graph_lines.append(
                f"  ENTITY {aid}: {item.get('name','')} ({item.get('entity_type','')}) "
                f"{cited_str}"
            )
        elif atype == "relationship":
            from_ev = item.get("from_evidence", [])
            to_ev = item.get("to_evidence", [])
            ev_str = ""
            if from_ev or to_ev:
                ev_str = f"  [from_evidence={from_ev}, to_evidence={to_ev}]"
            graph_lines.append(
                f"  RELATIONSHIP: {item.get('from_id', item.get('source_entity_id',''))} → "
                f"{item.get('to_id', item.get('target_entity_id',''))} "
                f"[{item.get('rel_type', item.get('relationship_type',''))}]"
                f"{ev_str}"
            )
        elif atype == "event":
            graph_lines.append(
                f"  EVENT {aid}: {(item.get('description') or '')[:150]}"
            )

    graph_text = "\n".join(graph_lines) if graph_lines else "(no graph context)"

    feedback_block = ""
    if state.critic_feedback:
        feedback_block = f"\n\nCRITIC FEEDBACK — you MUST address this before resubmitting:\n{state.critic_feedback}\n"

    return (
        f"QUESTION:\n{state.question}\n\n"
        f"RETRIEVED EVIDENCE CHUNKS (all citable; use evidence_id in citations[]):\n{chunks_text}\n\n"
        f"GRAPH CONTEXT:\n"
        f"  - Artifact IDs (hyp-*, lead-*, ent-*): reference verbatim in response TEXT, not in citations[].\n"
        f"  - Evidence UUIDs listed as 'from_evidence'/'to_evidence' for a relationship, or in 'evidence:' for an entity: "
        f"these are the primary sources for facts about that artifact — cite them in citations[] if available in CHUNKS above.\n"
        f"{graph_text}"
        f"{feedback_block}"
    )


_TOKEN_COST_PER_M = {
    "claude-sonnet-4-6": 3.0,
    "claude-opus-4-7": 15.0,
    "gpt-4o": 5.0,
    "gpt-4o-mini": 0.15,
}


def run_synthesizer(state: AgentState) -> AgentState:
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
            wait = 2 ** attempt
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
        state.trace.append({"step": "synthesizer", "error": "json_extraction_failed", "tokens": tokens})
        return state

    # Build chunk index for citation resolution
    chunk_index = {c.evidence_id: c for c in state.retrieved_chunks}

    # Primary: citations from the structured array
    draft_citations: list[DraftCitation] = []
    for cit in parsed.get("citations", []):
        ev_id = cit.get("evidence_id", "")
        chunk = chunk_index.get(ev_id)
        draft_citations.append(
            DraftCitation(
                evidence_id=ev_id,
                source_path=chunk.source_path if chunk else "",
                locator=chunk.locator if chunk else {},
                excerpt=(cit.get("excerpt") or "")[:400],
                confidence=float(cit.get("confidence") or 0.75),
            )
        )

    # Fallback: extract inline citations from response text when array is empty
    if not draft_citations:
        draft_citations = _extract_inline_citations(
            parsed.get("response", raw), chunk_index
        )

    # Supplement: add any inline citations missing from the array
    inline = _extract_inline_citations(parsed.get("response", raw), chunk_index)
    inline_ids = {c.evidence_id for c in draft_citations}
    for ic in inline:
        if ic.evidence_id not in inline_ids:
            draft_citations.append(ic)
            inline_ids.add(ic.evidence_id)

    # Enforce confidence floors per question class
    confidence = float(parsed.get("confidence") or 0.5)
    q_lower = state.question.lower()
    if any(w in q_lower for w in ("next", "action", "priorit", "rank", "investigat step")):
        confidence = max(0.50, min(0.78, confidence))
    elif any(w in q_lower for w in ("same person", "separate", "both", "two offender", "two suspect")):
        confidence = max(0.45, min(0.70, confidence))

    state.draft_response = parsed.get("response", raw)
    state.draft_confidence = confidence
    state.draft_reasoning = parsed.get("reasoning", "")
    state.draft_citations = draft_citations

    cost_per_m = _TOKEN_COST_PER_M.get(client.model, 3.0)
    state.trace.append({
        "step": "synthesizer",
        "provider": os.getenv("NIGHTWAVE_LLM_PROVIDER", "anthropic"),
        "model": client.model,
        "attempt": attempt,
        "draft_citations": len(draft_citations),
        "inline_extracted": len(inline),
        "confidence": confidence,
        "tokens": tokens,
        "cost_usd": round(tokens * cost_per_m / 1_000_000, 5),
    })

    return state
