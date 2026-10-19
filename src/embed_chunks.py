"""
Stage 2: Embed chunks with Gemini gemini-embedding-001.
Reads:  index/chunks.json
Writes: index/embeddings.npy, index/embed_cache.json (per-run checkpoint)

Run:  GEMINI_API_KEY=<key> python src/embed_chunks.py
"""

import json, hashlib, os, time
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types

CHUNKS_PATH = Path("index/chunks.json")
EMB_PATH    = Path("index/embeddings.npy")
CACHE_PATH  = Path("index/embed_cache.json")

MODEL      = "gemini-embedding-001"
DIMS       = 768
BATCH_SIZE = 20                    # max texts per request
MAX_BATCH_TOKENS = 8_000           # rough token ceiling per batch (chars / 4)
TARGET_TPM       = 15_000          # target tokens per minute (half the 30K limit)


def l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def embed_batch(client, texts: list[str]) -> list[list[float]]:
    """Embed one batch; retry up to 3× on 429/503; stop on daily quota."""
    for attempt in range(3):
        try:
            resp = client.models.embed_content(
                model=MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=DIMS,
                ),
            )
            return [e.values for e in resp.embeddings]
        except Exception as e:
            err = str(e)
            # Daily quota: repeated RESOURCE_EXHAUSTED after waiting won't help
            if "RESOURCE_EXHAUSTED" in err and attempt == 2:
                print(f"\n[DAILY QUOTA] {err[:200]}")
                raise SystemExit("Daily quota exhausted. Resume tomorrow.")
            if any(x in err for x in ("429", "503", "RESOURCE_EXHAUSTED", "SERVICE_UNAVAILABLE")):
                print(f"  [retry {attempt+1}] {err[:80]} — waiting 30 s")
                time.sleep(30)
            else:
                raise
    raise RuntimeError("All retries exhausted")


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running")
    client = genai.Client(api_key=api_key)

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    n = len(chunks)
    print(f"Loaded {n} chunks")

    # Load embed cache (keyed by sha256 of embed_text)
    cache: dict[str, list[float]] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    print(f"Cache: {len(cache)} entries")

    # Decide what needs embedding
    embeddings: list[list[float] | None] = [None] * n
    to_embed: list[tuple[int, str, str]] = []
    for i, c in enumerate(chunks):
        sha = hashlib.sha256(c["embed_text"].encode()).hexdigest()
        if sha in cache:
            embeddings[i] = cache[sha]
        else:
            to_embed.append((i, c["embed_text"], sha))

    from_cache = n - len(to_embed)
    print(f"To embed: {len(to_embed)}  |  from cache: {from_cache}")

    # Build batches: ≤ BATCH_SIZE texts AND ≤ MAX_BATCH_TOKENS estimated tokens
    batches: list[list[tuple]] = []
    batch, batch_tok = [], 0
    for item in to_embed:
        tok = len(item[1]) // 4
        if batch and (len(batch) >= BATCH_SIZE or batch_tok + tok > MAX_BATCH_TOKENS):
            batches.append(batch)
            batch, batch_tok = [], 0
        batch.append(item)
        batch_tok += tok
    if batch:
        batches.append(batch)

    print(f"Batches to send: {len(batches)}")

    # Rate-limit tracking: stay under TARGET_TPM
    window_tokens, window_start = 0, time.monotonic()

    for bi, batch in enumerate(batches):
        batch_tokens = sum(len(item[1]) // 4 for item in batch)
        window_tokens += batch_tokens

        elapsed = time.monotonic() - window_start
        if window_tokens > TARGET_TPM and elapsed < 60:
            wait = 61 - elapsed
            print(f"  Rate pause: {wait:.0f} s (window={window_tokens} tok)")
            time.sleep(wait)
            window_tokens, window_start = batch_tokens, time.monotonic()

        print(f"  Batch {bi+1}/{len(batches)}: {len(batch)} texts, ~{batch_tokens} tokens")
        vectors = embed_batch(client, [item[1] for item in batch])

        for (idx, _, sha), vec in zip(batch, vectors):
            normed = l2_norm(np.array(vec, dtype=np.float32)).tolist()
            embeddings[idx] = normed
            cache[sha] = normed

        # Save cache after every batch (resume-safe)
        CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")

    # Fill any remaining from cache (covers re-runs)
    for i, c in enumerate(chunks):
        if embeddings[i] is None:
            sha = hashlib.sha256(c["embed_text"].encode()).hexdigest()
            embeddings[i] = cache[sha]

    arr = np.array(embeddings, dtype=np.float32)
    np.save(str(EMB_PATH), arr)

    print(f"\nSaved {EMB_PATH}: shape={arr.shape}")
    print(f"  {'OK' if arr.shape[0] == n else 'FAIL'}: shape[0] == {n}")
    print(f"  Embedded now: {len(to_embed)}  |  from cache: {from_cache}")


if __name__ == "__main__":
    main()
