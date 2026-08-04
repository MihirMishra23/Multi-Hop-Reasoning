import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple


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
    Thin wrapper around FlashRAG's BM25 (bm25s backend) for per-example retrieval.
    Builds a tiny JSONL corpus and BM25 index in a temporary directory for the
    provided documents, queries it, and returns top-k 'title\\narticle' strings.
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

        # Lazily import heavy deps
        from flashrag.retriever.index_builder import Index_Builder
        from flashrag.retriever import BM25Retriever

        temp_dir = tempfile.mkdtemp(prefix="rag_bm25_")
        corpus_path = os.path.join(temp_dir, "corpus.jsonl")
        try:
            # Write JSONL corpus
            with open(corpus_path, "w", encoding="utf-8") as f:
                for i, contents in enumerate(contents_list):
                    json.dump({"id": i, "contents": contents}, f, ensure_ascii=False)
                    f.write("\n")

            # Build bm25s index
            index_builder = Index_Builder(
                retrieval_method="bm25",
                instruction=None,
                model_path=None,
                corpus_path=corpus_path,
                save_dir=temp_dir,
                max_length=0,
                batch_size=0,
                use_fp16=False,
                embedding_path=None,
                save_embedding=False,
                faiss_gpu=False,
                use_sentence_transformer=False,
                bm25_backend=self.bm25_backend,
            )
            index_builder.build_index()

            # Create retriever config and search
            config = {
                "retrieval_method": "bm25",
                "retrieval_topk": effective_top_k,
                "index_path": os.path.join(temp_dir, "bm25"),
                "corpus_path": corpus_path,
                "silent_retrieval": True,
                "bm25_backend": self.bm25_backend,
                # Cache/rerank off for this use-case
                "save_retrieval_cache": False,
                "retrieval_cache_path": "~",
                "use_retrieval_cache": False,
                "use_reranker": False,
            }
            retriever = BM25Retriever(config=config)
            results = _search_bm25(retriever, query, effective_top_k)

            # results may be dicts with 'contents' or id integers; normalize to contents strings
            if isinstance(results, list):
                return _map_results_to_docs(
                    results,
                    contents_list,
                    docs_list,
                )[:effective_top_k]
            return []
        finally:
            # Cleanup temporary directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


@dataclass
class FlashRAGBM25CorpusRetriever:
    """
    BM25 retriever that builds a corpus index once and reuses it across queries.
    """

    documents: List[Any]
    bm25_backend: str = "bm25s"

    def __post_init__(self) -> None:
        self._contents_list, self._docs_list = _build_contents_list(self.documents)
        self._temp_dir: Optional[str] = None
        self._corpus_path: Optional[str] = None
        self._retriever = None

        if not self._contents_list:
            return

        # Lazily import heavy deps
        from flashrag.retriever.index_builder import Index_Builder
        from flashrag.retriever import BM25Retriever

        self._temp_dir = tempfile.mkdtemp(prefix="rag_bm25_")
        self._corpus_path = os.path.join(self._temp_dir, "corpus.jsonl")

        # Write JSONL corpus
        with open(self._corpus_path, "w", encoding="utf-8") as f:
            for i, contents in enumerate(self._contents_list):
                json.dump({"id": i, "contents": contents}, f, ensure_ascii=False)
                f.write("\n")

        # Build bm25s index once
        index_builder = Index_Builder(
            retrieval_method="bm25",
            instruction=None,
            model_path=None,
            corpus_path=self._corpus_path,
            save_dir=self._temp_dir,
            max_length=0,
            batch_size=0,
            use_fp16=False,
            embedding_path=None,
            save_embedding=False,
            faiss_gpu=False,
            use_sentence_transformer=False,
            bm25_backend=self.bm25_backend,
        )
        index_builder.build_index()

        config = {
            "retrieval_method": "bm25",
            "retrieval_topk": max(1, min(1000, len(self._contents_list))),
            "index_path": os.path.join(self._temp_dir, "bm25"),
            "corpus_path": self._corpus_path,
            "silent_retrieval": True,
            "bm25_backend": self.bm25_backend,
            "save_retrieval_cache": False,
            "retrieval_cache_path": "~",
            "use_retrieval_cache": False,
            "use_reranker": False,
        }
        self._retriever = BM25Retriever(config=config)

    def retrieve(self, query: str, documents: List[Any], top_k: int) -> List[Any]:
        if not self._retriever:
            return []

        effective_top_k = _effective_top_k(top_k, len(self._contents_list))
        if effective_top_k == 0:
            return []

        results = _search_bm25(self._retriever, query, effective_top_k)
        if isinstance(results, list):
            return _map_results_to_docs(
                results,
                self._contents_list,
                self._docs_list,
            )[:effective_top_k]
        return []

    def close(self) -> None:
        if self._temp_dir:
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass
            self._temp_dir = None
            self._corpus_path = None

    def __del__(self) -> None:
        self.close()
