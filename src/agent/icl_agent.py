from typing import List

from src.agent.agent import Agent, LLM


def _join_evidence(docs: List[str]) -> str:
    return "\n================\n\n".join(docs)


class ICLAgent(Agent):
    """
    In-Context Learning Agent.
    - Includes all available context paragraphs (title + article) in the prompt
    - Prepends an 'Evidence' block to the prompt before the question
    """

    def __init__(
        self,
        llm: LLM,
        contexts: List[str],
        max_steps: int = 8,
    ) -> None:
        super().__init__(llm=llm, max_steps=max_steps)
        self.contexts = contexts or []

    def reset(self, contexts: List[str]) -> None:
        """Reset agent state for a new question with new contexts."""
        self.contexts = contexts or []
        self.trace = []

    def build_prompt(self, query: str) -> str:
        # Build step history (same style as base Agent)
        history = "\n".join(
            f"Step {i + 1} [{s.action}]: {s.answer or s.error or ''}".strip() for i, s in enumerate(self.trace)
        )
        # Include all contexts (no retrieval filtering)
        evidence_block = _join_evidence(self.contexts) if self.contexts else ""
        if history:
            return f"Evidence:\n{evidence_block}\n\n{history}\n\nQuestion: {query}".strip()
        else:
            return f"Evidence:\n{evidence_block}\n\nQuestion: {query}".strip()

