import ast
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]


class GrpoLauncherFlagTests(unittest.TestCase):
    def _run_launcher(self, *launcher_args):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()

            nvidia_smi = fake_bin / "nvidia-smi"
            nvidia_smi.write_text("#!/bin/sh\necho 'Fake GPU'\n")
            nvidia_smi.chmod(0o755)

            capture_path = temp_path / "accelerate-args.txt"
            accelerate = fake_bin / "accelerate"
            accelerate.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CAPTURE_FILE\"\n"
            )
            accelerate.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
            env["CAPTURE_FILE"] = str(capture_path)

            command = [
                "bash",
                str(REPO_ROOT / "scripts" / "grpo_train.sh"),
                "--model_path",
                "/models/Qwen3-1.7B",
                "--database_path",
                "/data/database.json",
                "--save_dir",
                str(temp_path / "checkpoints"),
                *launcher_args,
            ]
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return capture_path.read_text().splitlines(), result.stdout

    def test_explicit_retrieval_and_sampling_flags_are_forwarded_separately(self):
        launched_args, output = self._run_launcher(
            "--retrieval-top-k",
            "7",
            "--sampling-top-k",
            "23",
            "--retrieval-threshold",
            "0.75",
            "--adaptive-k",
            "--use-inverses",
        )

        self.assertIn("--retrieval_top_k=7", launched_args)
        self.assertIn("--retrieval_threshold=0.75", launched_args)
        self.assertIn("--top_k=23", launched_args)
        self.assertIn("--adaptive_k", launched_args)
        self.assertIn("--use_inverses", launched_args)
        self.assertIn("-rth0.75-rk7-sk23", output)
        self.assertIn("-inv", output)
        self.assertIn("Retrieval:   top-k=7, threshold=0.75, adaptive=True", output)
        self.assertIn("Inverses:    on", output)

    def test_legacy_top_k_keeps_sampling_behavior_and_now_configures_retrieval(self):
        launched_args, output = self._run_launcher("--top_k", "9")

        self.assertIn("--retrieval_top_k=9", launched_args)
        self.assertIn("--top_k=9", launched_args)
        self.assertIn("-rth0.6-rk9-sk9", output)

    def test_kebab_top_k_is_retrieval_only(self):
        launched_args, output = self._run_launcher("--top-k", "9")

        self.assertIn("--retrieval_top_k=9", launched_args)
        self.assertIn("--top_k=4", launched_args)
        self.assertIn("-rth0.6-rk9-sk4", output)

    def test_legacy_zero_top_k_disables_sampling_without_invalid_retrieval_k(self):
        launched_args, output = self._run_launcher("--top_k", "0")

        self.assertIn("--retrieval_top_k=1", launched_args)
        self.assertIn("--top_k=0", launched_args)
        self.assertIn("-rth0.6-rk1-sk0", output)

    def test_underscore_aliases_and_boolean_adaptive_value_are_supported(self):
        launched_args, output = self._run_launcher(
            "--retrieval_top_k",
            "6",
            "--retrieval_threshold",
            "0.8",
            "--use_adaptive_k",
            "true",
            "--use_inverses",
        )

        self.assertIn("--retrieval_top_k=6", launched_args)
        self.assertIn("--retrieval_threshold=0.8", launched_args)
        self.assertIn("--adaptive_k", launched_args)
        self.assertIn("--use_inverses", launched_args)
        self.assertIn("Retrieval:   top-k=6, threshold=0.8, adaptive=True", output)


class GrpoTrainerPlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer_tree = ast.parse(
            (REPO_ROOT / "src" / "trainer" / "lmlm_basetrainer.py").read_text()
        )
        cls.entrypoint_tree = ast.parse((REPO_ROOT / "src" / "grpo_train.py").read_text())

    @staticmethod
    def _keyword_attributes(call):
        return {
            keyword.arg: keyword.value.attr
            for keyword in call.keywords
            if keyword.arg is not None and isinstance(keyword.value, ast.Attribute)
        }

    def test_single_phase_database_load_receives_all_retrieval_settings(self):
        calls = [
            node
            for node in ast.walk(self.trainer_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load_database"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            self._keyword_attributes(calls[0]),
            {
                "top_k": "retrieval_top_k",
                "default_threshold": "retrieval_threshold",
                "adaptive": "adaptive_k",
                "use_inverses": "use_inverses",
            },
        )

    def test_two_phase_database_builder_receives_all_retrieval_settings(self):
        calls = [
            node
            for node in ast.walk(self.trainer_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_databases_from_triplets_batch"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            self._keyword_attributes(calls[0]),
            {
                "top_k": "retrieval_top_k",
                "default_threshold": "retrieval_threshold",
                "adaptive": "adaptive_k",
                "use_inverses": "use_inverses",
            },
        )

    def test_entrypoint_passes_parsed_flags_to_trainer(self):
        calls = [
            node
            for node in ast.walk(self.entrypoint_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "LMLMGRPOTrainer"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            {
                key: value
                for key, value in self._keyword_attributes(calls[0]).items()
                if key in {"retrieval_top_k", "retrieval_threshold", "adaptive_k", "use_inverses"}
            },
            {
                "retrieval_top_k": "retrieval_top_k",
                "retrieval_threshold": "retrieval_threshold",
                "adaptive_k": "adaptive_k",
                "use_inverses": "use_inverses",
            },
        )

    def test_direct_entrypoint_and_trainer_defaults_both_use_retrieval_k_one(self):
        lmlm_class = next(
            node
            for node in self.entrypoint_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LMLMArguments"
        )
        field_assignment = next(
            node
            for node in lmlm_class.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "retrieval_top_k"
        )
        dataclass_default = next(
            keyword.value
            for keyword in field_assignment.value.keywords
            if keyword.arg == "default"
        )

        trainer_class = next(
            node
            for node in self.trainer_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LMLMGRPOTrainer"
        )
        trainer_init = next(
            node
            for node in trainer_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        argument_names = [argument.arg for argument in trainer_init.args.args]
        trainer_defaults = dict(
            zip(argument_names[-len(trainer_init.args.defaults) :], trainer_init.args.defaults)
        )

        self.assertEqual(dataclass_default.value, 1)
        self.assertEqual(trainer_defaults["retrieval_top_k"].value, 1)


if __name__ == "__main__":
    unittest.main()
