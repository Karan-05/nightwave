"""Shared NLP utility functions for the multi-agent pipeline.

Extracted from retriever.py so they can be used by multiple subagents
without importing private symbols across module boundaries.
"""

from __future__ import annotations

import re


def extract_proper_nouns(text: str) -> list[str]:
    """Heuristic proper-noun extraction: consecutive Title-Case words."""
    # Match 1–3 consecutive Title-Cased words (names, platforms, locations)
    matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text)
    # Dedupe, drop stop-words, keep only plausible names (len > 2)
    stops = {
        "What",
        "Where",
        "When",
        "Who",
        "How",
        "Did",
        "Was",
        "The",
        "As",
        "Of",
        "On",
        "In",
        "For",
        "And",
        "Or",
        "At",
        "To",
        "A",
    }
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        if m not in stops and m not in seen:
            seen.add(m)
            result.append(m)
    return result[:6]


def extract_apps_platforms(text: str) -> list[str]:
    """Extract platform / app names: quoted words or known social-app patterns."""
    # Quoted single words/phrases
    quoted = re.findall(r'["\'`]([A-Za-z][\w\s]{0,20})["\']', text)
    # Unquoted but starts with capital + "app" nearby
    apps = re.findall(r"\b([A-Z][a-z]+)\s+app\b|\bapp\s+called\s+([A-Za-z]+)\b", text)
    flat = [a for group in apps for a in group if a]
    return list(dict.fromkeys(quoted + flat))[:4]
