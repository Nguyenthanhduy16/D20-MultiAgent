"""LangGraph workflow skeleton."""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
    ) -> None:
        self._supervisor = supervisor or SupervisorAgent()
        self._researcher = researcher or ResearcherAgent()
        self._analyst = analyst or AnalystAgent()
        self._writer = writer or WriterAgent()

    def build(self) -> CompiledStateGraph[ResearchState, None, ResearchState, ResearchState]:
        """Create a LangGraph graph.

        supervisor decides the next route by inspecting state; each worker runs once
        and hands control back to supervisor, which stops the graph once
        `state.route_history[-1] == "done"` (final_answer set, or max_iterations hit).
        """

        graph = StateGraph(ResearchState)

        graph.add_node("supervisor", self._supervisor.run)
        graph.add_node("researcher", self._researcher.run)
        graph.add_node("analyst", self._analyst.run)
        graph.add_node("writer", self._writer.run)

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state.route_history[-1],
            {
                ROUTE_RESEARCHER: "researcher",
                ROUTE_ANALYST: "analyst",
                ROUTE_WRITER: "writer",
                ROUTE_DONE: END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""

        compiled = self.build()
        result = compiled.invoke(state)
        return ResearchState.model_validate(result)
