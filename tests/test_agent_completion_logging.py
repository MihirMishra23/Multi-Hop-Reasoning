from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).parents[1]


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeLLM:
    def run(self, prompt, **kwargs):
        return FakeResponse("FINAL_ANSWER: Paris")


class FakeRetriever:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def retrieve(self, query, documents, top_k):
        self.calls.append((query, documents, top_k))
        return [{"title": "Paris", "contents": "Paris is the capital of France."}]


def _load_modules():
    llm_base = types.ModuleType("llm.base")
    llm_base.LLM = object
    llm_base.LLMResponse = FakeResponse
    llm_package = types.ModuleType("llm")

    agent_package = types.ModuleType("agent")
    agent_spec = importlib.util.spec_from_file_location(
        "agent.agent_class", ROOT / "src" / "agent" / "agent_class.py"
    )
    agent_class = importlib.util.module_from_spec(agent_spec)

    base_stubs = {
        "llm": llm_package,
        "llm.base": llm_base,
        "agent": agent_package,
        "agent.agent_class": agent_class,
    }
    with patch.dict(sys.modules, base_stubs):
        agent_spec.loader.exec_module(agent_class)

    retrieval = types.ModuleType("tools.retrieval")
    retrieval.BaseRetriever = FakeRetriever
    retrieval.FlashRAGBM25Retriever = FakeRetriever
    retrieval.FlashRAGBM25CorpusRetriever = FakeRetriever
    tools_package = types.ModuleType("tools")
    rag_spec = importlib.util.spec_from_file_location(
        "agent.rag_agent", ROOT / "src" / "agent" / "rag_agent.py"
    )
    rag_agent = importlib.util.module_from_spec(rag_spec)
    rag_stubs = {
        **base_stubs,
        "agent.agent_class": agent_class,
        "tools": tools_package,
        "tools.retrieval": retrieval,
        "agent.rag_agent": rag_agent,
    }
    with patch.dict(sys.modules, rag_stubs):
        rag_spec.loader.exec_module(rag_agent)
    return agent_class, rag_agent


agent_class, rag_agent = _load_modules()


class CompletionLoggingTests(unittest.TestCase):
    def test_base_agent_preserves_raw_model_completion(self):
        agent = agent_class.Agent(llm=FakeLLM(), max_steps=1)

        answers, traces = agent.run(["What is the capital of France?"])

        self.assertEqual(answers, ["Paris"])
        self.assertEqual(traces[0][-1].raw_response, "FINAL_ANSWER: Paris")

    def test_rag_retrieves_raw_question_and_adds_instruction_once(self):
        agent = rag_agent.RAGAgent(llm=FakeLLM(), corpus=["France: Paris"], max_steps=1)

        answers, traces = agent.run(["What is the capital of France?"])

        self.assertEqual(answers, ["Paris"])
        self.assertEqual(agent.retriever.calls[0][0], "What is the capital of France?")
        self.assertEqual(traces[0][-1].prompt.count("Provide only the final answer"), 1)
        self.assertNotIn("['What is the capital", traces[0][-1].prompt)
        self.assertEqual(traces[0][-1].raw_response, "FINAL_ANSWER: Paris")


if __name__ == "__main__":
    unittest.main()
