"""Index vectoriel FAISS en mémoire des chunks d'un CV (via LangChain)."""

import math

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.schemas import Chunk, SearchResult


class CVIndex:
    """Index FAISS en mémoire, construit pour une analyse et jamais persisté.

    Les vecteurs sont normalisés (norme L2 = 1) puis comparés par produit scalaire : le
    score renvoyé par `search` est donc une **similarité cosinus** dans [-1, 1], et non une
    distance. Plus il est élevé, plus le chunk est proche de la requête.
    """

    def __init__(self, chunks: list[Chunk], embeddings: Embeddings) -> None:
        if not chunks:
            raise ValueError("Impossible d'indexer un CV sans chunk.")
        self._size = len(chunks)
        self._store = FAISS.from_documents(
            [Document(page_content=_embedded_text(c), metadata=c.model_dump()) for c in chunks],
            _NormalizedEmbeddings(embeddings),
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

    @property
    def size(self) -> int:
        return self._size

    def search(self, query: str, k: int) -> list[SearchResult]:
        hits = self._store.similarity_search_with_score(query, k=min(k, self._size))
        return [SearchResult(chunk=Chunk.model_validate(doc.metadata), score=float(score)) for doc, score in hits]

    def search_many(self, queries: list[str], k: int) -> list[SearchResult]:
        """Une recherche par requête, fusionnées sans doublon : chaque chunk garde son
        meilleur score, puis les k meilleurs sont renvoyés (score décroissant)."""
        best: dict[int, SearchResult] = {}
        for query in dict.fromkeys(q.strip() for q in queries if q.strip()):
            for result in self.search(query, k):
                current = best.get(result.chunk.index)
                if current is None or result.score > current.score:
                    best[result.chunk.index] = result
        return sorted(best.values(), key=lambda r: r.score, reverse=True)[:k]


def _embedded_text(chunk: Chunk) -> str:
    # Le titre de section aide les requêtes comme "expérience React".
    return f"{chunk.section}\n{chunk.text}"


class _NormalizedEmbeddings(Embeddings):
    """Normalise les vecteurs : le produit scalaire FAISS devient une similarité cosinus,
    quel que soit le fournisseur."""

    def __init__(self, inner: Embeddings) -> None:
        self._inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_normalize(v) for v in self._inner.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return _normalize(self._inner.embed_query(text))


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector
