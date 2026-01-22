from typing import Any, Dict, List, Optional, Tuple

from agent.agent_class import Agent, AgentStep, LLM, LLMResponse
from tools.retrieval import BaseRetriever, FlashRAGBM25Retriever


def _join_evidence(docs: List[str]) -> str:
    return "\n================\n\n".join(docs)


class RAGAgent(Agent):
    """
    Retrieval-Augmented Agent.
    - Retrieves top-k context paragraphs (title + article) for a given question
    - Prepends an 'Evidence' block to the prompt before the question
    """

    def __init__(
        self,
        llm: LLM,
        retriever_type: str,
        contexts: List[str],
        rag_k: int = 4,
        max_steps: int = 8,
    ) -> None:
        super().__init__(llm=llm, max_steps=max_steps)
        # Initialize retriever by type
        match (retriever_type or "").lower():
            case "bm25":
                self.retriever: BaseRetriever = FlashRAGBM25Retriever()
            case _:
                raise NotImplementedError(f"Retriever type '{retriever_type}' is not implemented.")
        self.contexts = contexts or []
        self.rag_k = rag_k
        self._evidence_docs: List[str] = []

    def gather_evidence(self, query: str) -> None:
        """ Note that this is only computed once"""
        if self._evidence_docs:
            return
        self._evidence_docs = self.retriever.retrieve(query=query, documents=self.contexts, top_k=self.rag_k)

    def build_prompt(self, query: str) -> str:
        # Build step history (same style as base Agent)
        history = "\n".join(
            f"Step {i + 1} [{s.action}]: {s.answer or s.error or ''}".strip() for i, s in enumerate(self.trace)
        )
        evidence_block = _join_evidence(self._evidence_docs) if self._evidence_docs else ""
        if history:
            return f"Evidence:\n{evidence_block}\n\n{history}\n\nQuestion: {query}".strip()
        else:
            return f"Evidence:\n{evidence_block}\n\nQuestion: {query}".strip()

    def reset(self, contexts: List[str]) -> None:
        """Reset agent state for a new question with new contexts."""
        self.contexts = contexts or []
        self._evidence_docs = []
        self.trace = []


    # NOTE: We compute evidence once per run to avoid repeated indexing within multi-step loops
    def run(self, question: str, **llm_kwargs: Any) -> Tuple[Optional[str], List[AgentStep]]:
        query = self.build_query(question)
        self._evidence_docs = []
        self.gather_evidence(query)
        return super().run(query, **llm_kwargs)


