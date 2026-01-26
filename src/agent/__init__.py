"""Agent factory utilities."""

from typing import Any, Dict

from .agent import Agent
from .rag_agent import RAGAgent
from .icl_agent import ICLAgent
from .lmlm_agent import LMLMAgent


def get_agent(method: str, agent_kwargs: Dict[str, Any]) -> Agent:
    """Return an Agent instance for the given method.

    Args:
        method: Agent method type ("rag", "icl", "db", "lmlm")
        agent_kwargs: Dictionary containing agent-specific parameters

    Returns:
        Agent instance of the appropriate type

    Raises:
        NotImplementedError: For unsupported method types or configurations
        Exception: For missing required parameters
    """
    agent: Agent

    match method:
        case "icl":
            # Extract parameters
            dataset = agent_kwargs["dataset"]
            setting = agent_kwargs["setting"]
            max_steps = agent_kwargs.get("max_steps", 5)

            llm = agent_kwargs["llm"]

            # Guard against unsupported setting
            if dataset == "hotpotqa" and setting == "fullwiki":
                raise NotImplementedError("ICL is not supported for --setting fullwiki.")

            # Create ICL agent with empty contexts (will be reset per question)
            agent = ICLAgent(
                llm=llm,
                contexts=[],
                max_steps=max_steps,
            )

        case "rag":
            # Extract parameters
            retrieval = agent_kwargs.get("retrieval", "bm25")
            rag_k = agent_kwargs.get("rag_k", 4)
            max_steps = agent_kwargs.get("max_steps", 5)
            rag_corpus = agent_kwargs.get("rag_corpus")

            llm = agent_kwargs["llm"]

            # Guard against unsupported combinations (we only support bm25 + distractor for now)
            if retrieval != "bm25":
                raise NotImplementedError("Only --retrieval bm25 is supported currently.")

            # Create RAG agent with empty contexts (will be reset per question)
            agent = RAGAgent(
                llm=llm,
                retriever_type=retrieval,
                contexts=[],
                corpus=rag_corpus,
                rag_k=rag_k,
                max_steps=max_steps,
            )

        case "db":
            # Extract parameters
            max_steps = agent_kwargs.get("max_steps", 5)

            llm = agent_kwargs["llm"]

            agent = Agent(llm=llm, max_steps=max_steps)

        case "lmlm":
            # Extract parameters
            model_path = agent_kwargs.get("model_path")
            database_path = agent_kwargs.get("database_path")
            top_k = agent_kwargs.get("top_k")
            adaptive_k = agent_kwargs.get("adaptive_k")

            if model_path is None:
                raise Exception("You must set a local model path for lmlm setting")
            if database_path is None:
                raise Exception("You must set a local database path for lmlm setting")

            agent = LMLMAgent(model_path=model_path, database_path=database_path, adaptive = adaptive_k)

        case _:
            raise NotImplementedError(f"Method '{method}' is not implemented.")

    return agent


__all__ = ["Agent", "RAGAgent", "ICLAgent", "LMLMAgent", "get_agent"]
