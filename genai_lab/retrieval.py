"""Small, inspectable TF-IDF retrieval index for the project knowledge base."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_FILES = (
    "README.md",
    "PROJECT_WALKTHROUGH.md",
    "docs/GENAI_GLOSSARY.md",
    "agents.py",
    "environment.py",
    "run_evaluation.py",
    "app.py",
    "genai_lab/bandit_tools.py",
    "docs/GENAI_COURSE_ALIGNMENT.md",
    "docs/GENAI_MINI_PROJECT_REPORT.md",
)


@dataclass(frozen=True)
class Chunk:
    source: str
    title: str
    text: str
    chunk_number: int


def _window_text(text: str, size: int = 1100, overlap: int = 180) -> Iterable[str]:
    """Split text by characters while keeping a useful overlap between chunks."""
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        return
    start = 0
    length = len(normalized)
    while start < length:
        end = min(start + size, length)
        if end < length:
            boundary = normalized.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary
            else:
                boundary = normalized.rfind(" ", start + size // 2, end)
                if boundary > start:
                    end = boundary
        piece = normalized[start:end].strip()
        if piece:
            yield piece
        if end >= length:
            break
        start = max(start + 1, end - overlap)


class KnowledgeIndex:
    """Exact cosine search over an in-memory TF-IDF vector store.

    This is a transparent lexical baseline for a small corpus. It creates sparse
    term vectors, not neural semantic embeddings, and intentionally uses exact
    search instead of HNSW because the index is small and easy to inspect.
    """

    def __init__(self, root: Path = ROOT, file_paths: Iterable[str] = KNOWLEDGE_FILES):
        self.root = Path(root)
        self.chunks: List[Chunk] = []
        for relative_path in file_paths:
            path = self.root / relative_path
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            title = path.name
            for number, piece in enumerate(_window_text(text), start=1):
                self.chunks.append(
                    Chunk(
                        source=path.relative_to(self.root).as_posix(),
                        title=title,
                        text=piece,
                        chunk_number=number,
                    )
                )
        if not self.chunks:
            raise RuntimeError("The GenAI knowledge base has no readable source documents.")
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=50000,
        )
        self.matrix = self.vectorizer.fit_transform(chunk.text for chunk in self.chunks)

    def search(self, query: str, limit: int = 5):
        expanded = query
        lowered = query.lower()
        if any(term in lowered for term in ("reward", "like", "liked", "dislike", "no-like", "outcome")):
            expanded += (
                " A simulated viewer likes the selected movie means reward 1; no like means reward 0. "
                "The environment samples a Bernoulli outcome using the oracle p(like). "
                "This is not a recommendation accuracy label and does not mean the movie was the best action."
            )
        if "regret" in lowered:
            expanded += (
                " Average per-round difference between the best movie oracle probability and the chosen movie oracle probability. "
                "Lower regret means closer to the simulator oracle; it is not observed dislikes."
            )
        if "accurac" in lowered or "user wants" in lowered or "user preference" in lowered:
            expanded += (
                " This is a synthetic simulator, not a live viewer study. Rewards are sampled from a fixed oracle; "
                "no real user's current preferences or ground-truth recommendation accuracy are measured."
            )
        query_vector = self.vectorizer.transform([expanded])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        order = np.argsort(scores)[::-1]
        # Keep the strongest chunk per source so one large code file cannot
        # crowd all other evidence out of the context window.
        results = []
        seen_sources = set()
        for index in order:
            index = int(index)
            source = self.chunks[index].source
            if source in seen_sources or scores[index] <= 0:
                continue
            seen_sources.add(source)
            results.append({"chunk": self.chunks[index], "score": float(scores[index])})
            if len(results) >= max(1, min(limit, len(self.chunks))):
                break
        return results


knowledge_index = KnowledgeIndex()
