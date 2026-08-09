import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_retrieval_module():
    module_path = Path(__file__).parents[1] / "src" / "tools" / "retrieval.py"
    spec = importlib.util.spec_from_file_location("retrieval_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


retrieval = load_retrieval_module()


class FakeMemoryIndex:
    instances = []

    def __init__(self, contents, backend):
        self.contents = contents
        self.backend = backend
        self.search_calls = []
        self.__class__.instances.append(self)

    def search(self, query, top_k):
        self.search_calls.append((query, top_k))
        return list(range(top_k))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        FakeMemoryIndex.instances.clear()

    def test_effective_top_k_is_bounded_by_non_empty_documents(self):
        self.assertEqual(retrieval._effective_top_k(4, 2), 2)
        self.assertEqual(retrieval._effective_top_k(4, 10), 4)
        self.assertEqual(retrieval._effective_top_k(0, 10), 0)

    def test_per_example_retriever_clamps_bm25_search_depth(self):
        with patch.object(retrieval, "_BM25sMemoryIndex", FakeMemoryIndex):
            result = retrieval.FlashRAGBM25Retriever().retrieve(
                query="question",
                documents=["First: text", "Second: text", ""],
                top_k=4,
            )

        instance = FakeMemoryIndex.instances[-1]
        self.assertEqual(instance.contents, ["First\n\ntext", "Second\n\ntext"])
        self.assertEqual(instance.backend, "bm25s")
        self.assertEqual(instance.search_calls, [("question", 2)])
        self.assertEqual(result, ["First: text", "Second: text"])

    def test_corpus_retriever_builds_once_and_reuses_index(self):
        with patch.object(retrieval, "_BM25sMemoryIndex", FakeMemoryIndex):
            retriever = retrieval.FlashRAGBM25CorpusRetriever(
                ["First: text", "Second: text"]
            )
            first = retriever.retrieve("first question", [], 1)
            second = retriever.retrieve("second question", [], 1)

        self.assertEqual(len(FakeMemoryIndex.instances), 1)
        self.assertEqual(
            FakeMemoryIndex.instances[0].search_calls,
            [("first question", 1), ("second question", 1)],
        )
        self.assertEqual(first, ["First: text"])
        self.assertEqual(second, ["First: text"])

    def test_empty_tokenized_query_is_a_zero_hit_retrieval(self):
        class EmptyQueryRetriever:
            def search(self, query, num, return_score):
                raise ValueError(
                    "The query_tokens must be a list of list of tokens "
                    "(str for stemmed words, int for token ids matching corpus)"
                )

        self.assertEqual(
            retrieval._search_bm25(EmptyQueryRetriever(), "out of vocab", 4),
            [],
        )

    def test_unrelated_bm25_value_error_is_not_suppressed(self):
        class BrokenRetriever:
            def search(self, query, num, return_score):
                raise ValueError("corrupt index")

        with self.assertRaisesRegex(ValueError, "corrupt index"):
            retrieval._search_bm25(BrokenRetriever(), "question", 4)

    def test_memory_index_queries_with_corpus_vocabulary(self):
        corpus_calls = []
        retrieve_calls = []

        class FakeTokenizer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def tokenize(self, documents, **kwargs):
                corpus_calls.append((documents, kwargs))
                return ([['corpus-token']], {})

        class FakeBM25:
            def __init__(self, corpus, backend):
                self.corpus = corpus
                self.backend = backend

            def index(self, corpus_tokens, **kwargs):
                self.corpus_tokens = corpus_tokens

            def retrieve(self, query_tokens, **kwargs):
                retrieve_calls.append((query_tokens, kwargs))
                return ([[0]], None)

        fake_bm25s = types.ModuleType("bm25s")
        fake_bm25s.tokenization = types.SimpleNamespace(Tokenizer=FakeTokenizer)
        fake_bm25s.BM25 = FakeBM25

        def independent_query_tokenize(queries):
            raise AssertionError("query must not create an independent vocabulary")

        fake_bm25s.tokenize = independent_query_tokenize
        fake_stemmer = types.ModuleType("Stemmer")
        fake_stemmer.Stemmer = lambda language: f"stemmer:{language}"

        with patch.dict(sys.modules, {"bm25s": fake_bm25s, "Stemmer": fake_stemmer}):
            index = retrieval._BM25sMemoryIndex(["First document"])
            result = index.search("Question text", 1)

        self.assertEqual(result, [0])
        self.assertEqual(len(corpus_calls), 2)
        self.assertEqual(corpus_calls[1], (
            ["Question text"],
            {
                "update_vocab": False,
                "return_as": "tuple",
                "show_progress": False,
            },
        ))
        self.assertEqual(retrieve_calls[0][0], ([['corpus-token']], {}))


if __name__ == "__main__":
    unittest.main()
