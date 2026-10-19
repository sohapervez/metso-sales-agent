import sys
sys.path.insert(0, ".")
from src.search import search

queries = [
    "filter performance under changing process conditions slurry density cycle setpoints",
    "We have Larox pressure filters but performance drops whenever our feed grade changes",
    "optimize pressure filter cycle automatically moisture",
]

for q in queries:
    print(f"\nQuery: {q!r}")
    for r in search(q, k=5):
        lc = " [low_confidence]" if r.get("low_confidence") else ""
        model = r["model"] or "(text)"
        print(
            f"  {r['score']:.3f} (sim={r['similarity']:.3f} boost={r['boost']:.2f}) | "
            f"{r['brochure']} p{r['page']} | {model} | {r['text'][:100]!r}{lc}"
        )