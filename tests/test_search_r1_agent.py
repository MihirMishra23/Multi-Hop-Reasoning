import importlib.util
import json
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class FakeAgentStep:
    prompt: str
    answer: Optional[str]
    action: str
    error: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_result: Optional[Any] = None
    golden_triplets: Optional[str] = None


class FakeAgent:
    pass


def load_search_r1_module():
    module_path = Path(__file__).parents[1] / "src" / "agent" / "search_r1_agent.py"
    module_name = "search_r1_agent_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    agent_package = types.ModuleType("agent")
    agent_class = types.ModuleType("agent.agent_class")
    agent_class.Agent = FakeAgent
    agent_class.AgentStep = FakeAgentStep
    previous = {
        name: sys.modules.get(name) for name in ("agent", "agent.agent_class")
    }
    sys.modules["agent"] = agent_package
    sys.modules["agent.agent_class"] = agent_class
    try:
        spec.loader.exec_module(module)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module


search_r1 = load_search_r1_module()


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return f"<prompt>{messages[-1]['content']}</prompt><assistant>"

    @staticmethod
    def encode(text, add_special_tokens=False):
        return [ord(character) for character in text]

    @staticmethod
    def decode(token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeCompletion:
    def __init__(self, text):
        self.token_ids = [ord(character) for character in text]
        self.text = text


class FakeRequestOutput:
    def __init__(self, text):
        self.outputs = [FakeCompletion(text)]


class FakeEngine:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.prompts = []
        self.params = []

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.prompts.append(list(prompts))
        self.params.append(list(sampling_params))
        outputs = next(self.turns)
        return [FakeRequestOutput(text) for text in outputs]


class SearchR1ProtocolTests(unittest.TestCase):
    def test_parse_search_query_returns_last_match(self):
        # get_query() upstream keeps the last match because generation is
        # stopped at </search> and the tail is the query being asked for.
        text = (
            "<think>first</think><search>obsolete</search>"
            "<think>second</think><search>capital of France</search>"
        )
        self.assertEqual(search_r1.parse_search_query(text), "capital of France")

    def test_parse_search_query_none_when_no_search_tag(self):
        self.assertIsNone(search_r1.parse_search_query("<think>done</think>"))

    def test_format_search_response_matches_infer_py_document_format(self):
        # infer.py::_passages2string emits "Doc N(Title: T) body\n" with no
        # space before the paren, and no separator between docs. Wrapped in
        # <information>...</information>\n\n by curr_search_template.
        response = {
            "result": [
                [
                    {"document": {"contents": "Title A\nBody A"}},
                    {"document": {"contents": "Title B\nBody B"}},
                ]
            ]
        }
        self.assertEqual(
            search_r1.format_search_response(response),
            "<information>Doc 1(Title: Title A) Body A\n"
            "Doc 2(Title: Title B) Body B\n</information>\n\n",
        )

    def test_truncate_tool_response_preserves_information_tags(self):
        # Regression: truncation used to cut </information> off the end, so the
        # model spent its next turn balancing the tag instead of answering.
        long_body = "x" * 200
        text = f"<information>{long_body}</information>"
        result = search_r1.truncate_tool_response(text, limit=20, side="left")
        self.assertTrue(result.startswith("<information>"))
        self.assertTrue(result.endswith("</information>"))
        self.assertIn("...(truncated)", result)

    def test_multi_turn_run_splices_information_verbatim_into_prompt(self):
        # Turn 1: model emits <search>...</search>, retriever returns docs,
        # <information>...</information>\n\n is spliced in.
        # Turn 2: model emits <answer>Paris</answer>.
        turn1 = "<think>need it</think><search>capital of France</search>"
        turn2 = "<think>got it</think><answer>Paris</answer>"
        engine = FakeEngine([[turn1], [turn2]])
        agent = search_r1.SearchR1Agent(
            model_path="checkpoint",
            retrieval_url="http://retriever/retrieve",
            max_steps=3,
            max_model_len=10000,
            tokenizer=FakeTokenizer(),
            engine=engine,
            sampling_params_cls=FakeSamplingParams,
        )
        information = "<information>Doc 1(Title: Paris) capital of France\n</information>\n\n"
        agent._retrieve_many = lambda query_lists: [information]

        answers, traces = agent.run(["What is the capital of France?"], max_tokens=5000)

        self.assertEqual(answers, ["Paris"])
        self.assertEqual([step.action for step in traces[0]], ["toolcall", "finish"])
        step = traces[0][0]
        self.assertEqual(step.tool_name, "search")
        self.assertEqual(step.tool_args, {"query_list": ["capital of France"]})
        self.assertEqual(step.tool_result, information)
        # Second turn's prompt = initial prompt + '\n\n' + turn1 + information.
        # infer.py's curr_search_template prepends '\n\n' before every turn's
        # generated output; the <information> block is appended verbatim.
        second_prompt = engine.prompts[1][0]
        self.assertIn("\n\n" + turn1 + information, second_prompt)

    def test_infer_py_stop_variants_are_used(self):
        turn1 = "<think>need it</think><search>x</search>"
        turn2 = "<answer>done</answer>"
        engine = FakeEngine([[turn1], [turn2]])
        agent = search_r1.SearchR1Agent(
            model_path="checkpoint",
            retrieval_url="http://retriever/retrieve",
            max_steps=2,
            max_model_len=10000,
            tokenizer=FakeTokenizer(),
            engine=engine,
            sampling_params_cls=FakeSamplingParams,
        )
        agent._retrieve_many = lambda query_lists: ["<information>x</information>\n\n"]
        agent.run(["q?"], max_tokens=5000)

        self.assertEqual(
            engine.params[0][0].kwargs["stop"],
            [
                "</search>", " </search>",
                "</search>\n", " </search>\n",
                "</search>\n\n", " </search>\n\n",
            ],
        )
        self.assertTrue(engine.params[0][0].kwargs["include_stop_str_in_output"])


if __name__ == "__main__":
    unittest.main()
