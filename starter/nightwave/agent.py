"""Agent harness: single ReAct loop over the Madison Fields case.

Architecture: one LLM + 5 tools (search, entity, graph, citations, submit_answer).
The agent calls tools iteratively then terminates by calling submit_answer,
which forces a structured Answer. No regex parsing of free-form text.

Provider selection: LLM_PROVIDER env var (default: anthropic).
Thin adapter interface — one method: complete(messages, tools, system) -> dict.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from nightwave.tools import get_entity, list_citations, query_graph, search_evidence


@dataclass
class Citation:
    evidence_id: str
    source_path: str
    locator: dict
    excerpt: str
    confidence: float


@dataclass
class Step:
    kind: str  # "tool_call" | "tool_result" | "reasoning"
    detail: dict


@dataclass
class Answer:
    question: str
    response: str
    confidence: float  # 0-1, calibrated
    citations: list[Citation]
    trace: list[Step]


# ─── LLM adapter ─────────────────────────────────────────────────────────────


class LLMClient(ABC):
    @abstractmethod
    def complete(self, messages: list[dict], tools: list[dict], system: str) -> dict:
        """Return {"stop_reason": str, "content": list[dict]}"""
        ...


class AnthropicClient(LLMClient):
    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")

    def complete(self, messages: list[dict], tools: list[dict], system: str) -> dict:
        resp = self._client.messages.create(  # type: ignore[arg-type]
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
        )
        content = [blk.model_dump() for blk in resp.content]
        return {"stop_reason": resp.stop_reason, "content": content}


class OpenAIClient(LLMClient):
    """OpenAI adapter. Set OPENAI_API_KEY and LLM_PROVIDER=openai to use."""

    def __init__(self) -> None:
        from openai import OpenAI  # type: ignore[import-not-found]

        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o")

    def complete(self, messages: list[dict], tools: list[dict], system: str) -> dict:
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}] + messages,
            tools=oai_tools,
            max_tokens=4096,
        )
        choice = resp.choices[0]
        msg = choice.message
        content: list[dict] = []
        if msg.content:
            content.append({"type": "text", "text": msg.content})
        if msg.tool_calls:
            for tc in msg.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments),
                    }
                )
        stop_reason = "tool_use" if msg.tool_calls else "end_turn"
        return {"stop_reason": stop_reason, "content": content}


def get_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        return AnthropicClient()
    if provider == "openai":
        return OpenAIClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Supported: anthropic, openai")


# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a forensic investigative assistant for Nightwave,
working on the Madison Fields missing-person case.

CITATION RULES (non-negotiable):
- Every factual claim MUST be backed by a citation with a real evidence_id from the case.
- Use tools to retrieve evidence before making any claim.
- If no evidence exists for a claim, state that explicitly.
- Never fabricate evidence_ids or excerpts.

CONFIDENCE CALIBRATION — be honest about uncertainty:
- Single-source official documents (alert PDFs, DOJ press releases): 0.85–0.97
- Facts corroborated by 2+ independent sources: 0.80–0.95
- Reasoning over conflicting or ambiguous evidence: 0.45–0.70
- Prioritisation / planning where outcomes are uncertain: 0.50–0.78 (cap at 0.78)
- NEVER exceed 0.95 without definitive uncontested official documentation.
- NEVER exceed 0.70 on questions where evidence genuinely conflicts.
- For questions asking you to rank investigative actions (Q3-style), cap confidence at 0.78.

TOOL USE STRATEGY:
1. Start with search_evidence to find relevant raw text.
2. Use get_entity to pull the full record (with citations) for named suspects/victims/platforms.
3. Use query_graph for structural questions: leads, hypotheses, relationships, timelines.
4. Use list_citations when you need every citation for a specific artifact.
5. Call submit_answer once you have enough evidence — do not over-search.

When answering Q2-style questions (same person vs separate offenders), the case
hypotheses already encode the uncertainty. Reflect that in your confidence score.
When answering Q3-style questions (next actions), ground every action in the
existing leads and hypotheses — do not invent leads not in the case state.
"""

# ─── Tool schemas ─────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "search_evidence",
        "description": (
            "Full-text search over raw evidence files. Returns up to k hits with "
            "evidence_id, excerpt, locator, and confidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "k": {
                    "type": "integer",
                    "description": "Max results to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_entity",
        "description": (
            "Look up an entity by id (e.g. 'ent-kyle-lawrence') or name. "
            "Returns the full entity record with all citations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name_or_id": {"type": "string", "description": "Entity ID or display name"},
            },
            "required": ["name_or_id"],
        },
    },
    {
        "name": "query_graph",
        "description": (
            "Resolve structural questions against the case graph. "
            "Use for: 'Who is connected to Kyle Lawrence?', 'What are the active leads?', "
            "'What hypotheses link Madison and Josh?', 'List all events on Feb 13'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Structural question about the case graph",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "list_citations",
        "description": (
            "Return every citation pointing to an artifact (entity, event, lead, "
            "hypothesis) by its artifact_id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": (
                        "Artifact ID, e.g. 'ent-kyle-lawrence', "
                        "'hyp-josh-coconspirator'"
                    ),
                },
            },
            "required": ["artifact_id"],
        },
    },
    {
        "name": "submit_answer",
        "description": (
            "Submit your final, citation-backed answer. Call this once you have "
            "gathered sufficient evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": (
                        "Full answer text. Cite evidence inline as "
                        "[evidence_id: excerpt]."
                    ),
                },
                "confidence": {
                    "type": "number",
                    "description": (
                        "Calibrated confidence 0.0-1.0 per the calibration rules "
                        "in the system prompt."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Brief trace of what you found and how you reached your "
                        "conclusion."
                    ),
                },
                "citations": {
                    "type": "array",
                    "description": "All citations supporting this answer.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "evidence_id": {"type": "string"},
                            "source_path": {"type": "string"},
                            "locator": {"type": "object"},
                            "excerpt": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["evidence_id", "excerpt", "confidence"],
                    },
                },
            },
            "required": ["response", "confidence", "reasoning", "citations"],
        },
    },
]

# ─── Tool dispatcher ──────────────────────────────────────────────────────────


def _dispatch(name: str, inputs: dict) -> str:
    if name == "search_evidence":
        return json.dumps(search_evidence(inputs["query"], inputs.get("k", 5)), default=str)
    if name == "get_entity":
        return json.dumps(get_entity(inputs["name_or_id"]), default=str)
    if name == "query_graph":
        return json.dumps(query_graph(inputs["question"]), default=str)
    if name == "list_citations":
        return json.dumps(list_citations(inputs["artifact_id"]), default=str)
    if name == "submit_answer":
        return "SUBMIT"
    return json.dumps({"error": f"Unknown tool: {name}"})


# ─── ReAct loop ───────────────────────────────────────────────────────────────


def answer(question: str) -> Answer:
    """Answer `question` using case data. Returns a citation-backed Answer."""
    client = get_client()
    messages: list[dict] = [{"role": "user", "content": question}]
    trace: list[Step] = []
    final: Answer | None = None

    for _ in range(10):  # max iterations
        resp = client.complete(messages, TOOL_SCHEMAS, SYSTEM_PROMPT)
        content: list[dict] = resp.get("content", [])
        stop_reason: str = resp.get("stop_reason", "end_turn")

        messages.append({"role": "assistant", "content": content})

        tool_results: list[dict] = []
        submitted = False

        for block in content:
            btype = block.get("type")

            if btype == "text":
                trace.append(Step("reasoning", {"text": block.get("text", "")}))

            elif btype == "tool_use":
                tool_name: str = block["name"]
                tool_inputs: dict = block.get("input", {})
                tool_id: str = block.get("id", "")
                trace.append(Step("tool_call", {"tool": tool_name, "inputs": tool_inputs}))

                if tool_name == "submit_answer":
                    submitted = True
                    final = Answer(
                        question=question,
                        response=tool_inputs.get("response", ""),
                        confidence=float(tool_inputs.get("confidence", 0.5)),
                        citations=[
                            Citation(
                                evidence_id=c.get("evidence_id", ""),
                                source_path=c.get("source_path", ""),
                                locator=c.get("locator", {}),
                                excerpt=c.get("excerpt", ""),
                                confidence=float(c.get("confidence", 0.5)),
                            )
                            for c in tool_inputs.get("citations", [])
                        ],
                        trace=trace,
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": "Answer recorded.",
                        }
                    )
                else:
                    raw = _dispatch(tool_name, tool_inputs)
                    trace.append(Step("tool_result", {"tool": tool_name, "result": raw[:600]}))
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": tool_id, "content": raw}
                    )

        if submitted and final is not None:
            break

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        elif stop_reason == "end_turn":
            # Model stopped without calling submit_answer — best-effort extraction
            text = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
            final = Answer(
                question=question,
                response=text or "No answer produced.",
                confidence=0.3,
                citations=[],
                trace=trace,
            )
            break

    if final is None:
        final = Answer(
            question=question,
            response="Max iterations reached without a final answer.",
            confidence=0.1,
            citations=[],
            trace=trace,
        )

    return final
