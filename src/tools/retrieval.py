import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Protocol


class BaseRetriever(Protocol):
    def retrieve(self, query: str, documents: List[str], top_k: int) -> List[str]:
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


def _build_contents_list(documents: List[str]) -> List[str]:
    contents_list: List[str] = []
    for ctx in documents or []:
        title, article = _split_title_article(str(ctx))
        contents_list.append(f"{title}\n{article}".strip())
    return contents_list


@dataclass
class FlashRAGBM25Retriever:
    """
    Thin wrapper around FlashRAG's BM25 (bm25s backend) for per-example retrieval.
    Builds a tiny JSONL corpus and BM25 index in a temporary directory for the
    provided documents, queries it, and returns top-k 'title\\narticle' strings.
    """

    bm25_backend: str = "bm25s"

    def retrieve(self, query: str, documents: List[str], top_k: int) -> List[str]:
        # Prepare contents list as '{title}\\n{article}'
        contents_list = _build_contents_list(documents)

        if len(contents_list) == 0:
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
                "retrieval_topk": top_k,
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
            results = retriever.search(query=query, num=top_k, return_score=False)

            # results may be dicts with 'contents' or id integers; normalize to contents strings
            top_contents: List[str] = []
            if isinstance(results, list) and len(results) > 0:
                first = results[0]
                if isinstance(first, dict) and "contents" in first:
                    top_contents = [str(item["contents"]) for item in results]
                else:
                    # Assume results are integer IDs into the corpus
                    top_contents = [contents_list[int(idx)] for idx in results]
            return top_contents[:top_k]
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

    documents: List[str]
    bm25_backend: str = "bm25s"

    def __post_init__(self) -> None:
        self._contents_list = _build_contents_list(self.documents)
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

    def retrieve(self, query: str, documents: List[str], top_k: int) -> List[str]:
        if not self._retriever:
            return []

        results = self._retriever.search(query=query, num=top_k, return_score=False)
        top_contents: List[str] = []
        if isinstance(results, list) and len(results) > 0:
            first = results[0]
            if isinstance(first, dict) and "contents" in first:
                top_contents = [str(item["contents"]) for item in results]
            else:
                top_contents = [self._contents_list[int(idx)] for idx in results]
        return top_contents[:top_k]

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
