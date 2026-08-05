import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class FakeCuda:
    @staticmethod
    def is_available():
        return False


class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.encode_calls = []

    def encode(self, texts, **kwargs):
        self.encode_calls.append((list(texts), kwargs))
        return np.arange(len(texts) * 2, dtype=np.float32).reshape(len(texts), 2)


class FakeTopkRetriever:
    def __init__(
        self,
        database,
        model_name=None,
        top_k=5,
        adaptive=False,
        threshold=0.6,
        *args,
        **kwargs,
    ):
        self.database = list(database)
        self.top_k = top_k
        self.is_adaptive = adaptive
        self.precomputed_embeddings = kwargs.get("precomputed_embeddings")
        self.use_inverses = kwargs.get("use_inverses")
        self.default_threshold = threshold

    def retrieve_top_k(self, entity, relationship, threshold=None, return_triplets=False):
        matches = [
            triplet
            for triplet in self.database
            if triplet[0] == entity and triplet[1] == relationship
        ][: self.top_k]
        if return_triplets:
            return [f"({entity}, {relationship}, {value})" for _, _, value in matches]
        return [value for _, _, value in matches]

    @staticmethod
    def _normalize_text(text):
        return text.strip().lower()


def load_database_manager_module():
    """Load database_manager with lightweight stand-ins for optional ML dependencies."""
    module_path = (
        Path(__file__).parents[1]
        / "src"
        / "multi_lmlm"
        / "database"
        / "database_manager.py"
    )
    module_name = "database_manager_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    torch_module = types.ModuleType("torch")
    torch_module.cuda = FakeCuda()
    datasets_module = types.ModuleType("datasets")
    datasets_module.DatasetDict = type("DatasetDict", (), {})
    sentence_transformers_module = types.ModuleType("sentence_transformers")
    sentence_transformers_module.SentenceTransformer = FakeSentenceTransformer
    retriever_module = types.ModuleType("multi_lmlm.database.topk_retriever")
    retriever_module.TopkRetriever = FakeTopkRetriever

    dependency_stubs = {
        "torch": torch_module,
        "datasets": datasets_module,
        "sentence_transformers": sentence_transformers_module,
        "multi_lmlm.database.topk_retriever": retriever_module,
    }
    with patch.dict(sys.modules, dependency_stubs):
        spec.loader.exec_module(module)
    return module


database_manager = load_database_manager_module()


class InverseTripletTests(unittest.TestCase):
    def test_add_inverse_triplets_preserves_order_and_deduplicates(self):
        triplets = [
            ("Alice", "parent", "Bob"),
            ("Alice", "parent", "Bob"),
            ("Carol", "identity", "Carol"),
        ]

        augmented = database_manager._add_inverse_triplets(triplets)

        self.assertEqual(
            augmented,
            [
                ("Alice", "parent", "Bob"),
                ("Carol", "identity", "Carol"),
                ("Bob", "parent", "Alice"),
            ],
        )

    def test_non_batched_builder_adds_inverse_triplets(self):
        manager = database_manager.DatabaseManager()

        manager.build_database_from_triplets(
            [("Alice", "parent", "Bob")], use_inverses=True
        )

        self.assertEqual(
            manager.database["triplets"],
            [("Alice", "parent", "Bob"), ("Bob", "parent", "Alice")],
        )
        self.assertEqual(manager.topk_retriever.database, manager.database["triplets"])
        self.assertFalse(manager.topk_retriever.use_inverses)

    def test_disabled_flag_preserves_original_triplets(self):
        triplets = [
            ("Alice", "parent", "Bob"),
            ("Alice", "parent", "Bob"),
        ]
        manager = database_manager.DatabaseManager()

        manager.build_database_from_triplets(triplets, use_inverses=False)

        self.assertEqual(manager.database["triplets"], triplets)

    def test_batched_builder_embeds_and_splits_augmented_triplets(self):
        managers = database_manager.build_databases_from_triplets_batch(
            [
                [("Alice", "parent", "Bob")],
                [("Paris", "country", "France")],
            ],
            top_k=3,
            default_threshold=0.25,
            adaptive=True,
            use_inverses=True,
        )

        self.assertEqual(
            managers[0].database["triplets"],
            [("Alice", "parent", "Bob"), ("Bob", "parent", "Alice")],
        )
        self.assertEqual(
            managers[1].database["triplets"],
            [("Paris", "country", "France"), ("France", "country", "Paris")],
        )
        np.testing.assert_array_equal(
            managers[0].topk_retriever.precomputed_embeddings,
            np.array([[0, 1], [2, 3]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            managers[1].topk_retriever.precomputed_embeddings,
            np.array([[4, 5], [6, 7]], dtype=np.float32),
        )
        self.assertTrue(all(manager.topk_retriever.top_k == 3 for manager in managers))
        self.assertTrue(
            all(manager.topk_retriever.default_threshold == 0.25 for manager in managers)
        )
        self.assertTrue(all(manager.topk_retriever.is_adaptive for manager in managers))
        self.assertEqual(
            managers[0].retrieve_from_database(
                "<|db_entity|>Bob<|db_relationship|>parent<|db_return|>"
            ),
            ["Alice"],
        )

    def test_empty_batch_does_not_initialize_a_retriever(self):
        managers = database_manager.build_databases_from_triplets_batch(
            [[], []], use_inverses=True
        )

        self.assertEqual(len(managers), 2)
        self.assertTrue(all(manager.database["triplets"] == set() for manager in managers))
        self.assertTrue(all(manager.topk_retriever is None for manager in managers))

    def test_precomputed_builder_embeds_only_missing_inverses(self):
        manager = database_manager.DatabaseManager()
        embedding_model = FakeSentenceTransformer()

        manager.build_database_from_triplets_with_embeddings(
            [("Alice", "parent", "Bob")],
            np.array([[10.0, 11.0]], dtype=np.float32),
            use_inverses=True,
            embedding_model=embedding_model,
        )

        self.assertEqual(
            manager.database["triplets"],
            [("Alice", "parent", "Bob"), ("Bob", "parent", "Alice")],
        )
        np.testing.assert_array_equal(
            manager.topk_retriever.precomputed_embeddings,
            np.array([[10.0, 11.0], [0.0, 1.0]], dtype=np.float32),
        )
        self.assertEqual(embedding_model.encode_calls[0][0], ["bob parent"])

    def test_precomputed_builder_rejects_misaligned_embeddings(self):
        manager = database_manager.DatabaseManager()

        with self.assertRaisesRegex(ValueError, "one embedding per triplet"):
            manager.build_database_from_triplets_with_embeddings(
                [("Alice", "parent", "Bob")],
                np.empty((0, 2), dtype=np.float32),
            )

    def test_json_loader_adds_inverse_triplets_and_updates_metadata(self):
        data = {
            "entities": ["Alice"],
            "relationships": ["parent"],
            "return_values": ["Bob"],
            "triplets": [["Alice", "parent", "Bob"]],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "database.json"
            database_path.write_text(json.dumps(data))
            manager = database_manager.DatabaseManager()

            manager.load_database(
                str(database_path),
                top_k=7,
                default_threshold=0.25,
                adaptive=True,
                use_inverses=True,
            )

        self.assertEqual(
            manager.database["triplets"],
            {("Alice", "parent", "Bob"), ("Bob", "parent", "Alice")},
        )
        self.assertEqual(manager.database["entities"], {"Alice", "Bob"})
        self.assertEqual(manager.database["return_values"], {"Alice", "Bob"})
        self.assertEqual(manager.topk_retriever.top_k, 7)
        self.assertEqual(manager.topk_retriever.default_threshold, 0.25)
        self.assertTrue(manager.topk_retriever.is_adaptive)
        self.assertFalse(manager.topk_retriever.use_inverses)
        self.assertEqual(
            manager.retrieve_from_database(
                "<|db_entity|>Bob<|db_relationship|>parent<|db_return|>"
            ),
            ["Alice"],
        )


if __name__ == "__main__":
    unittest.main()
