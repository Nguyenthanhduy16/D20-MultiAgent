"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

SYSTEM_PROMPT = (
    "You are the Researcher agent in a multi-agent research pipeline. "
    "You receive raw search results and must summarize them into concise, factual "
    "research notes for the Analyst agent to work from. Stick to what the sources "
    "actually say, do not speculate, and reference sources by their [n] index."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._search_client = search_client or SearchClient()
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        sources = self._search_client.search(
            state.request.query, max_results=state.request.max_sources
        )
        state.sources = sources

        sources_block = "\n".join(
            f"[{i}] {source.title} ({source.url or 'no url'}): {source.snippet}"
            for i, source in enumerate(sources, start=1)
        ) or "(no sources found)"

        user_prompt = (
            f"Research question: {state.request.query}\n\n"
            f"Search results:\n{sources_block}\n\n"
            "Task: write concise research notes summarizing what these sources say "
            "about the question. If no sources were found, say so explicitly."
        )

        response = self._llm_client.complete(SYSTEM_PROMPT, user_prompt)
        state.research_notes = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        )
        state.add_trace_event(
            "researcher",
            {
                "sources_found": len(sources),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )

        return state
