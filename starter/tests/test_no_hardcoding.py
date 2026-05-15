"""Sweep test — asserts no case-specific strings are hardcoded in subagent routing logic.

These strings were previously baked into classify/routing code, which would break
for any case other than Nightwave. They belong in data (case_data.json) or in
system prompts as LLM guidance, not in Python keyword dispatch tables.
"""

from __future__ import annotations

import ast
from pathlib import Path

SUBAGENTS_DIR = Path(__file__).resolve().parents[1] / "nightwave" / "multiagent" / "subagents"

FORBIDDEN_ENTITY_IDS = [
    "ent-kyle-lawrence",
    "ent-josh",
]

FORBIDDEN_CYPHER_ARGS = {"madison", "fields"}

# Platform names that must not appear as members of Python tuple/list
# dispatch tables (like multi_hop_terms or memory routing lists).
# They ARE allowed in system-prompt strings (LLM instructional text).
FORBIDDEN_DISPATCH_KEYWORDS = {"snapchat", "session"}


def _subagent_files() -> list[Path]:
    return [p for p in SUBAGENTS_DIR.glob("*.py") if not p.name.startswith("__")]


def _string_literals_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _tuple_list_members(path: Path) -> list[str]:
    """Return string values that appear as direct elements of tuple/list literals."""
    tree = ast.parse(path.read_text())
    members: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    members.append(elt.value)
    return members


def _call_args(path: Path) -> list[tuple[str, str]]:
    """Return (func_name, first_string_arg) for all calls in file."""
    tree = ast.parse(path.read_text())
    results: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else (func.id if isinstance(func, ast.Name) else "")
        )
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            results.append((name, node.args[0].value))
    return results


# ── Entity IDs must not be hardcoded ─────────────────────────────────────────


def test_no_hardcoded_entity_ids() -> None:
    """Entity IDs like 'ent-kyle-lawrence' must not be hardcoded in subagent code."""
    for path in _subagent_files():
        for lit in _string_literals_in_file(path):
            for forbidden in FORBIDDEN_ENTITY_IDS:
                assert forbidden not in lit, (
                    f"{path.name}: hardcoded entity ID '{forbidden}' — "
                    "derive entity IDs dynamically from question proper nouns."
                )


# ── Forbidden _cypher_entity_context args ─────────────────────────────────────


def test_no_hardcoded_cypher_entity_names() -> None:
    """_cypher_entity_context('madison') / ('fields') must not appear in subagent code."""
    for path in _subagent_files():
        for func_name, arg_val in _call_args(path):
            if func_name in ("_cypher_entity_context", "_cypher_entity_with_citations"):
                assert arg_val.lower() not in FORBIDDEN_CYPHER_ARGS, (
                    f"{path.name}: hardcoded arg '{arg_val}' in {func_name}() — "
                    "extract entity names from the question text instead."
                )


# ── Platform keywords must not be in dispatch tuples/lists ────────────────────


def test_no_platform_names_in_dispatch_lists() -> None:
    """Platform names must not appear as members of Python tuple/list dispatch tables.

    LLM system-prompt strings may contain platform names; this test checks only
    tuple/list literal members (e.g., multi_hop_terms = (..., 'snapchat', ...)).
    """
    for path in _subagent_files():
        for member in _tuple_list_members(path):
            for kw in FORBIDDEN_DISPATCH_KEYWORDS:
                assert kw not in member.lower(), (
                    f"{path.name}: platform keyword '{kw}' found in a tuple/list literal — "
                    "platform names belong in case data or LLM prompts, not code dispatch."
                )
