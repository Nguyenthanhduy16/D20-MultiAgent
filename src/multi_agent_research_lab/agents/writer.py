"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

SYSTEM_PROMPT = (
    "You are the Writer agent in a multi-agent research pipeline. "
    "You receive research notes, analysis notes, and their sources, and must "
    "synthesize a clear, well-structured final answer. Every non-obvious claim "
    "must cite a source by its [n] index. Do not introduce claims that are not "
    "backed by the notes or sources."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        if not state.analysis_notes:
            raise ValidationError("WriterAgent requires state.analysis_notes to be set")

        sources_block = "\n".join(
            f"[{i}] {source.title} ({source.url or 'no url'})"
            for i, source in enumerate(state.sources, start=1)
        ) or "(no sources attached)"

        user_prompt = (
            f"Research question: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            f"Analysis notes:\n{state.analysis_notes}\n\n"
            f"Sources:\n{sources_block}\n\n"
            "Task: write the final answer to the research question for this audience, "
            "citing sources by their [n] index."
        )

        response = self._llm_client.complete(SYSTEM_PROMPT, user_prompt)
        state.final_answer = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        )
        state.add_trace_event(
            "writer",
            {
                "sources_cited": len(state.sources),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )

        return state
