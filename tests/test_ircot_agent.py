from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from unittest.mock import patch


@dataclass
class FakeAgentStep:
    prompt: str
    answer: str | None
    action: str
    raw_response: str | None = None
    error: str | None = None
    tool_name: str | None = None
    tool_args: dict | None = None
    golden_triplets: str | None = None


class FakeAgent:
    def __init__(self, llm, max_steps=8, **kwargs):
        self.llm = llm
        self.max_steps = max_steps
        self.trace = []


class FakeLLMBase:
    pass


def load_ircot_module():
    module_path = Path(__file__).parents[1] / "src" / "agent" / "ircot_agent.py"
    spec = importlib.util.spec_from_file_location("ircot_agent_under_test", module_path)
    module = importlib.util.module_from_spec(spec)

    agent_class = types.ModuleType("agent.agent_class")
    agent_class.Agent = FakeAgent
    agent_class.AgentStep = FakeAgentStep
    agent_class.LLM = FakeLLMBase

    ircot_constants = types.ModuleType("agent.ircot_constants")
    ircot_constants.OFFICIAL_IRCOT_COMMIT = "3c1820f698eea5eeddb4fba3c56b64c961e063e4"
    ircot_constants.OFFICIAL_IRCOT_URL = "https://github.com/StonyBrookNLP/ircot"
    ircot_constants.OFFICIAL_CORPUS_NAMES = {
        "hotpotqa": "hotpotqa",
        "2wiki": "2wikimultihopqa",
        "musique": "musique",
    }

    requests = types.ModuleType("requests")
    requests.RequestException = RuntimeError
    requests.post = None

    rapidfuzz = types.ModuleType("rapidfuzz")
    rapidfuzz.fuzz = types.SimpleNamespace(
        ratio=lambda left, right: 100 * SequenceMatcher(None, left, right).ratio()
    )
    stubs = {
        "agent": types.ModuleType("agent"),
        "agent.agent_class": agent_class,
        "agent.ircot_constants": ircot_constants,
        "requests": requests,
        "rapidfuzz": rapidfuzz,
    }
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


ircot = load_ircot_module()
ircot._token_length = lambda text, tokenizer_model_name="gpt2": len(text.split())


@dataclass
class FakeResponse:
    text: str


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def run(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return FakeResponse(self.responses.pop(0))


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, documents, top_k):
        self.queries.append((query, list(documents), top_k))
        if len(self.queries) == 1:
            return [
                {
                    "title": "France",
                    "paragraph_text": "France is a country in Europe.",
                    "corpus_name": "hotpotqa",
                },
                {
                    "title": "Paris",
                    "paragraph_text": "Paris is the capital of France.",
                    "corpus_name": "hotpotqa",
                },
            ]
        return [
            {
                "title": "Paris",
                "paragraph_text": "Paris is the capital of France.",
                "corpus_name": "hotpotqa",
            },
            {
                "title": "Paris history",
                "paragraph_text": "Paris has a long history.",
                "corpus_name": "hotpotqa",
            },
        ]


def official_prompt_file():
    prompt = (
        '# METADATA: {"qid": "5ab92dba554299131ca422a2"}\n'
        "Wikipedia Title: France\nFrance is in Europe.\n\n"
        "Q: What is the capital of France?\n"
        "A: France's capital is Paris. So the answer is: Paris.\n"
    )
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    handle.write(prompt)
    handle.close()
    return handle.name


class IRCoTAgentTests(unittest.TestCase):
    def test_official_loop_retrieves_reasons_then_runs_fresh_reader(self):
        llm = FakeLLM(
            [
                "France's capital city is Paris. Extra text is ignored.",
                "So the answer is: Paris.",
                "France's capital is Paris. So the answer is: Paris.",
            ]
        )
        retriever = FakeRetriever()
        agent = ircot.IRCoTAgent(
            llm=llm,
            dataset="hotpotqa",
            prompt_file=official_prompt_file(),
            retriever=retriever,
            retrieval_k=6,
            max_evidence=15,
            max_steps=10,
            generator_max_tokens=300,
            sentence_segmenter=lambda text: text.split(" Extra", 1)[0],
            verify_prompt_hash=False,
        )

        answers, traces = agent.run("What is the capital of France?", temperature=0.0)

        self.assertEqual(answers, ["Paris"])
        self.assertEqual(len(traces[0]), 3)
        self.assertEqual(retriever.queries[0], ("the capital of France?", [], 6))
        self.assertEqual(retriever.queries[1][0], "France's capital city Paris.")
        self.assertEqual(len(agent._evidence_docs), 3)
        self.assertEqual(len(agent._retrieval_rounds), 2)
        self.assertEqual(agent._retrieval_rounds[1]["cumulative_evidence_count"], 3)
        self.assertEqual(llm.calls[0][1]["max_tokens"], 300)
        self.assertEqual(llm.calls[0][1]["stop"], ["\n"])
        self.assertEqual(llm.calls[0][1]["extra"]["max_input_tokens"], 8000)
        self.assertNotIn("France's capital city is Paris", llm.calls[-1][0])
        self.assertTrue(llm.calls[-1][0].endswith("Q: What is the capital of France?\nA:"))

    def test_reasoning_starters_and_arithmetic_match_official_rules(self):
        self.assertTrue(ircot.is_reasoning_sentence("Therefore the author follows."))
        self.assertTrue(ircot.is_reasoning_sentence("The difference is 9 - 4 = 5."))
        self.assertFalse(ircot.is_reasoning_sentence("The novel was published in 1965."))
        query = ircot.retrieval_query(
            "Who wrote the novel?",
            ["The novel was published in 1965.", "Therefore, the author can be identified."],
        )
        self.assertEqual(query, "The novel was published in 1965.")

    def test_released_prompt_sets_use_the_exact_upstream_qids(self):
        prompt = (
            '# METADATA: {"qid": "5ab92dba554299131ca422a2"}\n'
            "SET ONE\n"
            '# METADATA: {"qid": "5a88f9d55542995153361218"}\n'
            "SET TWO\n"
            '# METADATA: {"qid": "not-selected"}\n'
            "NEITHER\n"
        )
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        handle.write(prompt)
        handle.close()

        set_one = ircot.read_official_prompt(handle.name, "hotpotqa", prompt_set="1")
        set_two = ircot.read_official_prompt(handle.name, "hotpotqa", prompt_set="2")

        self.assertEqual(set_one, "SET ONE")
        self.assertEqual(set_two, "SET TWO")
        with self.assertRaisesRegex(ValueError, "1, 2, or 3"):
            ircot.read_official_prompt(handle.name, "hotpotqa", prompt_set="4")

    def test_fuzzy_title_and_paragraph_duplicate_is_removed(self):
        self.assertTrue(
            ircot.is_para_closely_matching(
                ["Paris"],
                ["Paris is the capital city of France."],
                "Paris ",
                "Paris is the capital city of France!",
            )
        )

    def test_final_answer_extraction_matches_official_fallbacks(self):
        self.assertEqual(ircot.extract_official_answer("So the answer is: Paris."), "Paris")
        self.assertEqual(ircot.extract_official_answer('"Paris"'), "Paris")
        self.assertEqual(ircot.extract_official_answer("Paris"), "Paris")

    def test_official_retriever_request_schema(self):
        class Response:
            ok = True
            status_code = 200

            @staticmethod
            def json():
                return {"retrieval": [{"title": "Paris", "paragraph_text": "Text"}]}

        calls = []

        def fake_post(url, json, timeout):
            calls.append((url, json, timeout))
            return Response()

        retriever = ircot.OfficialIRCoTRetriever(
            "http://retriever:8000", "hotpotqa", retry_delay_seconds=0
        )
        with patch.object(ircot.requests, "post", fake_post):
            result = retriever.retrieve("capital France", [], 6)

        self.assertEqual(result[0]["title"], "Paris")
        self.assertEqual(calls[0][0], "http://retriever:8000/retrieve")
        self.assertEqual(
            calls[0][1],
            {
                "retrieval_method": "retrieve_from_elasticsearch",
                "query_text": "capital France",
                "max_hits_count": 6,
                "corpus_name": "hotpotqa",
                "document_type": "title_paragraph_text",
            },
        )

    def test_parameter_validation(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            ircot.IRCoTAgent(
                llm=FakeLLM([]),
                dataset="hotpotqa",
                prompt_file=official_prompt_file(),
                retriever=FakeRetriever(),
                retrieval_k=0,
                verify_prompt_hash=False,
            )


if __name__ == "__main__":
    unittest.main()
