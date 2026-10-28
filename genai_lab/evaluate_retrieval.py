"""Evaluate the RAG retriever against the small labeled query set.

Run: .venv/bin/python -m genai_lab.evaluate_retrieval
This evaluates retrieval only; it does not call or score an LLM.
"""

import json
from pathlib import Path

from genai_lab.retrieval import ROOT, knowledge_index


def main():
    cases_path = Path(__file__).with_name("evaluation_queries.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    reciprocal_ranks = []
    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    rows = []

    for case in cases:
        hits = knowledge_index.search(case["query"], limit=5)
        ranked_sources = [hit["chunk"].source for hit in hits]
        relevant = set(case["relevant_sources"])
        first_rank = next(
            (rank for rank, source in enumerate(ranked_sources, start=1) if source in relevant),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        recall_at_1.append(float(any(source in relevant for source in ranked_sources[:1])))
        recall_at_3.append(float(any(source in relevant for source in ranked_sources[:3])))
        recall_at_5.append(float(any(source in relevant for source in ranked_sources[:5])))
        rows.append((case["query"], first_rank, ranked_sources[:3]))

    print("RAG retrieval evaluation (source-level relevance; manually labeled)")
    print(f"Queries: {len(cases)}")
    print(f"Recall@1: {sum(recall_at_1) / len(cases):.3f}")
    print(f"Recall@3: {sum(recall_at_3) / len(cases):.3f}")
    print(f"Recall@5: {sum(recall_at_5) / len(cases):.3f}")
    print(f"MRR:      {sum(reciprocal_ranks) / len(cases):.3f}")
    print("\nPer-query first relevant rank and top 3 sources:")
    for query, rank, sources in rows:
        print(f"- rank={rank or 'miss'} | {query}\n  {', '.join(sources)}")
    print("\nCorpus root:", ROOT)


if __name__ == "__main__":
    main()
