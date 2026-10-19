"""
Stage 3: Semantic search over embedded chunks with keyword boost.

Standalone use:  GEMINI_API_KEY=<key> python src/search.py
Library use:     from src.search import search
"""

import json, os, re
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types

CHUNKS_PATH  = Path("index/chunks.json")
EMB_PATH     = Path("index/embeddings.npy")
ALIASES_PATH = Path("index/aliases.json")

MODEL = "gemini-embedding-001"
DIMS  = 768

# Boost constants (tunable)
BOOST_MODEL  = 0.10   # exact model alias appears in query
BOOST_PAIR   = 0.10   # (series alias, number) pair both appear in query
BOOST_FAMILY = 0.0    # family alias of chunk's brochure appears in query
MIN_SCORE    = 0.55   # below this: flag low_confidence in results


# ---- Lazy-loaded globals (loaded once on first search call) ----
_chunks  = None
_emb     = None
_family  = None
_client  = None


def _load():
    global _chunks, _emb, _family, _client
    if _chunks is not None:
        return
    _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    _emb    = np.load(str(EMB_PATH))       # float32 [n_chunks, 768]
    _family = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))["family"]
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY before searching")
    _client = genai.Client(api_key=api_key)


def l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _norm_q(q: str) -> str:
    """Lowercase; replace hyphens with spaces so 'PF-DS' matches 'pf ds'."""
    return re.sub(r'-', ' ', q.lower())


def _keyword_boost(chunk: dict, norm_q: str) -> float:
    tokens = set(re.findall(r'\w+', norm_q))
    boost  = 0.0

    # +BOOST_MODEL: chunk's model alias appears as whole token(s)
    for alias in chunk.get("aliases", []):
        atoks = set(re.findall(r'\w+', _norm_q(alias)))
        if atoks and atoks <= tokens:
            boost += BOOST_MODEL
            break

    # +BOOST_PAIR: series alias AND number both appear as tokens
    for series, number in chunk.get("pairs", []):
        stoks = set(re.findall(r'\w+', _norm_q(series)))
        if stoks <= tokens and number in tokens:
            boost += BOOST_PAIR
            break

    # +BOOST_FAMILY: family alias of the brochure appears as whole token(s)
    for alias in _family.get(chunk.get("brochure", ""), []):
        atoks = set(re.findall(r'\w+', _norm_q(alias)))
        if atoks and atoks <= tokens:
            boost += BOOST_FAMILY
            break

    return boost


def search(query: str, k: int = 5) -> list[dict]:
    """
    Embed query, compute cosine similarity + keyword boost, return top-k chunks.
    Each result dict: score, similarity, boost, brochure, page, model, id, text.
    Adds low_confidence=True when best score < MIN_SCORE.
    """
    _load()

    # Embed query with RETRIEVAL_QUERY task type
    resp = _client.models.embed_content(
        model=MODEL,
        contents=[query],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=DIMS,
        ),
    )
    q_vec = l2_norm(np.array(resp.embeddings[0].values, dtype=np.float32))

    # Cosine similarity (stored vectors are L2-normalised → dot product = cosine)
    sims   = _emb @ q_vec
    norm_q = _norm_q(query)
    boosts = np.array([_keyword_boost(c, norm_q) for c in _chunks], dtype=np.float32)
    finals = sims + boosts

    top_idx = np.argsort(finals)[::-1][:k]
    low_flag = bool(finals[top_idx[0]] < MIN_SCORE)

    results = []
    for i in top_idx:
        c = _chunks[i]
        entry = {
            "score":      float(finals[i]),
            "similarity": float(sims[i]),
            "boost":      float(boosts[i]),
            "brochure":   c["brochure"],
            "page":       c["page"],
            "model":      c["model"],
            "id":         c["id"],
            "text":       c["text"],
        }
        if low_flag:
            entry["low_confidence"] = True
        results.append(entry)
    return results


# ============================================================
# CLI test
# ============================================================

if __name__ == "__main__":
    TEST_QUERIES = [
        "We have Larox pressure filters but performance drops whenever our feed grade changes",
        "AC 2000 max service flow",
        "PF-DS 18 chambers 25 mm chamber volume",
        "which pump for highly abrasive mill discharge slurry",
        "Larox PF 60 filter height with 120 m2 filtration area",
        "what is the price of the AC 2000",
    ]

    for q in TEST_QUERIES:
        print(f"\nQuery: {q!r}")
        for r in search(q):
            lc = " [low_confidence]" if r.get("low_confidence") else ""
            model_str = r["model"] or "(text)"
            print(f"  {r['score']:.3f} | {r['brochure']} p{r['page']} | "
                  f"{model_str} | {r['text'][:100]}{lc}")
