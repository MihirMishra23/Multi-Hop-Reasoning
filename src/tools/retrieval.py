import logging
from dataclasses import dataclass
from typing import Any, List, Protocol, Tuple


class BaseRetriever(Protocol):
    def retrieve(self, query: str, documents: List[Any], top_k: int) -> List[Any]:
        ...


def _split_title_article(context: str) -> tuple[str, str]:
    """
    HotpotQA contexts are formatted as 'Title: sentence sentence ...'.
    Split once on ': ' to get title and article. If no delimiter is found,
    treat the whole string as article with empty title.
    """
    if ": " in context:
        title, article = context.split(": ", 1)
        return title.strip(), article.strip()
    return "", context.strip()


def _normalize_document(doc: Any) -> Tuple[str, Any]:
    """Return (index_text, original_doc) for a document string or structured record."""
    if isinstance(doc, dict):
        title = str(doc.get("title", "")).strip()
        contents = (
            doc.get("contents")
            if doc.get("contents") is not None
            else doc.get("context")
        )
        if contents is None:
            contents = doc.get("paragraph_text", "")
        contents_text = str(contents).strip()
        combined = f"{title}\n\n{contents_text}".strip()
        return combined, doc
    title, article = _split_title_article(str(doc))
    combined = f"{title}\n\n{article}".strip()
    return combined, str(doc)


def _build_contents_list(documents: List[Any]) -> Tuple[List[str], List[Any]]:
    """Build indexable text list and aligned original-doc list for mapping results."""
    contents_list: List[str] = []
    docs_list: List[Any] = []
    for doc in documents or []:
        combined, original = _normalize_document(doc)
        if not combined:
            continue
        contents_list.append(combined)
        docs_list.append(original)
    return contents_list, docs_list


def _effective_top_k(top_k: int, document_count: int) -> int:
    """Clamp retrieval depth to the non-empty documents in the index.

    ``bm25s==0.2.1`` raises from its numba backend when ``k`` exceeds the
    number of indexed documents instead of returning the shorter result.  A
    few Table 3 examples contain fewer than four non-empty contexts, so the
    evaluator must handle this at the wrapper boundary.
    """
    return max(0, min(int(top_k), int(document_count)))


def _search_bm25(retriever: Any, query: str, top_k: int) -> List[Any]:
    """Search BM25, treating an out-of-vocabulary query as no retrieval.

    The pinned FlashRAG/bm25s stack raises ``ValueError`` when every query
    token is absent from the index vocabulary.  That is a valid zero-hit
    retrieval round, especially for model-generated IRCoT queries.  Preserve
    all other errors so corpus or index failures remain visible.
    """
    try:
        return retriever.search(query=query, num=top_k, return_score=False)
    except ValueError as exc:
        if "query_tokens must be a list of list of tokens" not in str(exc):
            raise
        logging.getLogger(__name__).warning(
            "BM25 query contained no indexed tokens; returning no evidence"
        )
        return []


class _BM25sMemoryIndex:
    """In-memory BM25s index compatible with the cluster's BM25s releases.

    FlashRAG's on-disk index builder calls tokenizer serialization methods that
    are absent from BM25s 0.2.0.  RAG evaluation does not need persistence: the
    per-example indexes are tiny, and the PopQA corpus index lives for the
    lifetime of one evaluator process.  Building and querying the same BM25s
    backend directly avoids that version-sensitive serialization boundary.
    """

    def __init__(self, contents: List[str], backend: str = "bm25s") -> None:
        if backend != "bm25s":
            raise ValueError(f"Unsupported BM25 backend: {backend}")

        import bm25s
        import Stemmer

        self._tokenizer = bm25s.tokenization.Tokenizer(
            stopwords="en",
            stemmer=Stemmer.Stemmer("english"),
        )
        corpus_tokens = self._tokenizer.tokenize(
            contents,
            return_as="tuple",
            show_progress=False,
        )
        self._retriever = bm25s.BM25(
            corpus=list(range(len(contents))),
            backend="numba",
        )
        self._retriever.index(corpus_tokens, show_progress=False)

    def search(self, query: str, top_k: int) -> List[int]:
        query_tokens = self._tokenizer.tokenize(
            [query],
            update_vocab=False,
            return_as="tuple",
            show_progress=False,
        )
        try:
            retrieved = self._retriever.retrieve(
                query_tokens,
                k=top_k,
                show_progress=False,
            )
        except ValueError as exc:
            if "query_tokens must be a list of list of tokens" not in str(exc):
                raise
            logging.getLogger(__name__).warning(
                "BM25 query contained no indexed tokens; returning no evidence"
            )
            return []

        documents = getattr(retrieved, "documents", retrieved[0])
        if hasattr(documents, "tolist"):
            documents = documents.tolist()
        if documents and isinstance(documents[0], list):
            documents = documents[0]
        return [int(doc_id) for doc_id in documents]


def _map_results_to_docs(
    results: List[Any],
    contents_list: List[str],
    docs_list: List[Any],
) -> List[Any]:
    """Map retriever outputs back to original docs (prefer IDs when available)."""
    if not results:
        return []
    first = results[0]
    if isinstance(first, dict):
        if "id" in first:
            return [docs_list[int(item["id"])] for item in results]
        if "contents" in first:
            lookup = {contents: i for i, contents in enumerate(contents_list)}
            mapped: List[Any] = []
            for item in results:
                contents = str(item.get("contents", ""))
                idx = lookup.get(contents)
                if idx is None:
                    mapped.append(contents)
                else:
                    mapped.append(docs_list[idx])
            return mapped
    return [docs_list[int(idx)] for idx in results]


@dataclass
class FlashRAGBM25Retriever:
    """
    BM25s retriever for per-example retrieval.

    The public class name is retained for compatibility with existing configs.
    """

    bm25_backend: str = "bm25s"

    def retrieve(self, query: str, documents: List[Any], top_k: int) -> List[Any]:
        # Prepare contents list as '{title}\\n\\n{article}'
        contents_list, docs_list = _build_contents_list(documents)

        if len(contents_list) == 0:
            return []

        effective_top_k = _effective_top_k(top_k, len(contents_list))
        if effective_top_k == 0:
            return []

        index = _BM25sMemoryIndex(contents_list, backend=self.bm25_backend)
        results = index.search(query, effective_top_k)
        return _map_results_to_docs(results, contents_list, docs_list)[
            :effective_top_k
        ]


@dataclass
class FlashRAGBM25CorpusRetriever:
    """
    BM25 retriever that builds a corpus index once and reuses it across queries.
    """

    documents: List[Any]
    bm25_backend: str = "bm25s"

    def __post_init__(self) -> None:
        self._contents_list, self._docs_list = _build_contents_list(self.documents)
        self._retriever = None

        if not self._contents_list:
            return

        self._retriever = _BM25sMemoryIndex(
            self._contents_list,
            backend=self.bm25_backend,
        )

    def retrieve(self, query: str, documents: List[Any], top_k: int) -> List[Any]:
        if not self._retriever:
            return []

        effective_top_k = _effective_top_k(top_k, len(self._contents_list))
        if effective_top_k == 0:
            return []

        results = self._retriever.search(query, effective_top_k)
        return _map_results_to_docs(
            results,
            self._contents_list,
            self._docs_list,
        )[:effective_top_k]

    def close(self) -> None:
        self._retriever = None

    def __del__(self) -> None:
        self.close()
