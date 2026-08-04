import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


@dataclass
class FakeAgentStep:
    prompt: str
    answer: str | None
    action: str
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

    retrieval = types.ModuleType("tools.retrieval")
    retrieval.BaseRetriever = object
    retrieval.FlashRAGBM25CorpusRetriever = object
    retrieval.FlashRAGBM25Retriever = object

    stubs = {
        "agent": types.ModuleType("agent"),
        "agent.agent_class": agent_class,
        "tools": types.ModuleType("tools"),
        "tools.retrieval": retrieval,
    }
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


ircot = load_ircot_module()


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
                {"title": "France", "contents": "France is a country in Europe."},
                {"title": "Paris", "contents": "Paris is the capital of France."},
            ]
        return [
            {"title": "Paris", "contents": "Paris is the capital of France."},
            {"title": "Paris history", "contents": "Paris has a long history."},
        ]


class IRCoTAgentTests(unittest.TestCase):
    def test_interleaves_retrieval_and_reasoning_and_saves_rounds(self):
        llm = FakeLLM(
            [
                "France's capital city is Paris. Extra text should be ignored.",
                "FINAL_ANSWER: Paris",
                "FINAL_ANSWER: Paris",
            ]
        )
        retriever = FakeRetriever()
        contexts = ["France context", "Paris context"]
        agent = ircot.IRCoTAgent(
            llm=llm,
            contexts=contexts,
            retriever=retriever,
            retrieval_k=6,
            max_evidence=15,
            max_steps=8,
            step_max_tokens=48,
        )

        answers, traces = agent.run(
            "What is the capital of France?",
            temperature=0.0,
            max_tokens=128,
        )

        self.assertEqual(answers, ["Paris"])
        self.assertEqual(len(traces[0]), 3)
        self.assertEqual(retriever.queries[0], ("the capital of France?", contexts, 6))
        self.assertEqual(retriever.queries[1][0], "France's capital city Paris.")
        self.assertEqual(len(agent._evidence_docs), 3)
        self.assertEqual(len(agent._retrieval_rounds), 2)
        self.assertEqual(agent._retrieval_rounds[1]["cumulative_evidence_count"], 3)
        self.assertEqual(llm.calls[0][1]["max_tokens"], 48)
        self.assertEqual(llm.calls[-1][1]["max_tokens"], 128)

    def test_reasoning_only_sentence_falls_back_to_last_factual_thought(self):
        query = ircot._retrieval_query(
            "Who wrote the novel?",
            ["The novel was published in 1965.", "Therefore, the author can be identified."],
        )

        self.assertEqual(query, "The novel was published in 1965.")

    def test_parameter_validation(self):
        with self.assertRaisesRegex(ValueError, "retrieval_k"):
            ircot.IRCoTAgent(llm=FakeLLM([]), retriever=FakeRetriever(), retrieval_k=0)


if __name__ == "__main__":
    unittest.main()
