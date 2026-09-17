"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

SYSTEM_PROMPT = (
    "You are the Analyst agent in a multi-agent research pipeline. "
    "You receive research notes and their sources, and must extract key claims, "
    "compare differing viewpoints across sources, and flag claims that rely on weak "
    "or single-source evidence. Be concise and structured. Reference sources by "
    "their [n] index when relevant."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes` from `state.research_notes` and `state.sources`."""

        if not state.research_notes:
            raise ValidationError("AnalystAgent requires state.research_notes to be set")

        sources_block = "\n".join(
            f"[{i}] {source.title} ({source.url or 'no url'}): {source.snippet}"
            for i, source in enumerate(state.sources, start=1)
        ) or "(no sources attached)"

        user_prompt = (
            f"Research question: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            f"Sources:\n{sources_block}\n\n"
            "Task: extract the key claims, compare viewpoints across sources, and flag "
            "any claims with weak or single-source evidence."
        )

        response = self._llm_client.complete(SYSTEM_PROMPT, user_prompt)
        state.analysis_notes = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        )
        state.add_trace_event(
            "analyst",
            {
                "sources_used": len(state.sources),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )

        return state
