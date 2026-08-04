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


class FakeIndexBuilder:
    def __init__(self, **kwargs):
        pass

    def build_index(self):
        pass


class FakeBM25Retriever:
    instances = []

    def __init__(self, config):
        self.config = config
        self.search_calls = []
        self.__class__.instances.append(self)

    def search(self, query, num, return_score):
        self.search_calls.append((query, num, return_score))
        return list(range(num))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        FakeBM25Retriever.instances.clear()

    def test_effective_top_k_is_bounded_by_non_empty_documents(self):
        self.assertEqual(retrieval._effective_top_k(4, 2), 2)
        self.assertEqual(retrieval._effective_top_k(4, 10), 4)
        self.assertEqual(retrieval._effective_top_k(0, 10), 0)

    def test_per_example_retriever_clamps_bm25_search_depth(self):
        index_builder = types.ModuleType("flashrag.retriever.index_builder")
        index_builder.Index_Builder = FakeIndexBuilder
        flashrag_retriever = types.ModuleType("flashrag.retriever")
        flashrag_retriever.BM25Retriever = FakeBM25Retriever
        stubs = {
            "flashrag": types.ModuleType("flashrag"),
            "flashrag.retriever": flashrag_retriever,
            "flashrag.retriever.index_builder": index_builder,
        }

        with patch.dict(sys.modules, stubs):
            result = retrieval.FlashRAGBM25Retriever().retrieve(
                query="question",
                documents=["First: text", "Second: text", ""],
                top_k=4,
            )

        instance = FakeBM25Retriever.instances[-1]
        self.assertEqual(instance.config["retrieval_topk"], 2)
        self.assertEqual(instance.search_calls, [("question", 2, False)])
        self.assertEqual(result, ["First: text", "Second: text"])


if __name__ == "__main__":
    unittest.main()
