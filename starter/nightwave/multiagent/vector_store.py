"""Hybrid retrieval: BM25 + dense vector embeddings with RRF re-ranking.

Model: all-MiniLM-L6-v2 (22M params, 384-dim, fast, no GPU needed).
Index is built once at module load from the same corpus as tools._INDEX.
RRF constant k=60 (standard, balances BM25-heavy vs vector-heavy queries).

No external vector DB needed — cosine similarity over numpy matrix is
fast enough for <1000 chunks at <1ms query time.
"""

from __future__ import annotations

import os
import time
from typing import Any

from nightwave.tools import _INDEX, _bm25_score, _tokenize

_RRF_K = 60
_MODEL_NAME = "all-MiniLM-L6-v2"

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

_embeddings = None  # shape (N, 384)
_model = None
_corpus_snapshot: list[dict] = []
_dense_disabled_reason: str | None = None


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        print(f"[vector_store] Loading {_MODEL_NAME}…")
        t0 = time.time()
        _model = SentenceTransformer(_MODEL_NAME)
        print(f"[vector_store] Model loaded in {time.time() - t0:.1f}s")
    return _model


def build_index() -> None:
    """Embed the full corpus. Called lazily on first query."""
    global _dense_disabled_reason, _embeddings, _corpus_snapshot
    if _dense_disabled_reason is not None:
        raise RuntimeError(_dense_disabled_reason)
    if os.getenv("NIGHTWAVE_DISABLE_DENSE", "").lower() in {"1", "true", "yes"}:
        _dense_disabled_reason = "dense retrieval disabled by NIGHTWAVE_DISABLE_DENSE"
        raise RuntimeError(_dense_disabled_reason)
    if _embeddings is not None:
        return
    if not _HAS_NUMPY:
        raise RuntimeError("numpy unavailable")

    model = _load_model()
    _corpus_snapshot = list(_INDEX.corpus)
    texts = [d["text"] for d in _corpus_snapshot]

    print(f"[vector_store] Embedding {len(texts)} chunks…")
    t0 = time.time()
    vecs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    _embeddings = np.array(vecs, dtype=np.float32)
    print(f"[vector_store] Index ready in {time.time() - t0:.1f}s  shape={_embeddings.shape}")


def _vector_scores(query: str) -> list[tuple[int, float]]:
    """Return (corpus_idx, cosine_sim) sorted descending."""
    build_index()
    model = _load_model()
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    sims = (_embeddings @ q_vec).tolist()  # cosine sim (embeddings are L2-normed)
    return sorted(enumerate(sims), key=lambda x: x[1], reverse=True)


def _bm25_scores(query: str, case_id: str | None = None) -> list[tuple[int, float]]:
    """Return (corpus_idx, bm25_score) sorted descending."""
    q_tokens = _tokenize(query)
    active_case_id = case_id or _INDEX.case_id
    scored = [
        (i, _bm25_score(q_tokens, doc))
        for i, doc in enumerate(_INDEX.corpus)
        if doc.get("case_id") == active_case_id
    ]
    return sorted((item for item in scored if item[1] > 0), key=lambda x: x[1], reverse=True)


def _rrf(rankings: list[list[tuple[int, float]]], k: int = _RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists of (idx, score)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query: str, k: int = 8, case_id: str | None = None) -> list[dict[str, Any]]:
    """BM25 + vector with RRF. Returns up to k deduplicated hits."""
    active_case_id = case_id or _INDEX.case_id
    bm25_ranked = _bm25_scores(query, active_case_id)
    try:
        build_index()
        vec_ranked = _vector_scores(query)
        rankings = [bm25_ranked, vec_ranked]
        source = "rrf"
    except Exception as exc:
        global _dense_disabled_reason
        reason = str(exc)
        first_failure = _dense_disabled_reason != reason
        _dense_disabled_reason = reason
        if first_failure:
            print(f"[vector_store] Dense retrieval unavailable ({reason}); using BM25 only")
        if not bm25_ranked:
            return []
        rankings = [bm25_ranked]
        source = "bm25"

    fused = _rrf(rankings)

    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for idx, rrf_score in fused[: k * 2]:  # oversample then trim
        doc = _INDEX.corpus[idx]
        if doc.get("case_id") != active_case_id:
            continue
        ev_id = doc["evidence_id"]
        key = f"{ev_id}:{doc['locator']}"
        if key in seen_ids:
            continue
        seen_ids.add(key)

        ev = _INDEX.evidence_by_id.get(ev_id, {})
        results.append(
            {
                "evidence_id": ev_id,
                "case_id": active_case_id,
                "filename": doc["filename"],
                "excerpt": doc["text"][:400],
                "locator": doc["locator"],
                "rrf_score": round(rrf_score, 6),
                "source": source,
                "evidence_name": ev.get("evidence_name", ""),
                "evidence_confidence": ev.get("confidence", 0.5),
            }
        )

        if len(results) >= k:
            break

    return results
