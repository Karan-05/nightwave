"""CaseStore read-model tests."""

from __future__ import annotations

import pytest

from nightwave.case import current_case_id
from nightwave.case_store import CaseIndex, JsonCaseStore


def test_json_case_store_loads_active_case_by_id() -> None:
    store = JsonCaseStore()
    case_id = current_case_id()

    case = store.load_case(case_id)

    assert case["case_id"] == case_id


def test_json_case_store_rejects_unknown_case_id() -> None:
    store = JsonCaseStore()

    with pytest.raises(KeyError):
        store.load_case("wrong-case")


def test_case_index_builds_case_scoped_corpus() -> None:
    index = CaseIndex()

    assert index.case_id == current_case_id()
    assert index.corpus
    assert {doc["case_id"] for doc in index.corpus} == {index.case_id}
    assert index.entities_by_id
    assert index.evidence_by_id
