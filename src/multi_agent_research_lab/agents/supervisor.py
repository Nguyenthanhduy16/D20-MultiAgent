"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Append the next route to `state.route_history`.

        Routing policy, based on what is still missing in state:
        - no `research_notes` yet -> researcher
        - `research_notes` but no `analysis_notes` -> analyst
        - `analysis_notes` but no `final_answer` -> writer
        - `final_answer` set, or `max_iterations` reached -> done

        `graph/workflow.py` should read `state.route_history[-1]` to decide which
        node to run next, and stop the graph once it is "done".
        """

        settings = get_settings()

        if state.final_answer:
            route = ROUTE_DONE
        elif state.iteration >= settings.max_iterations:
            state.errors.append(
                f"Stopped after {state.iteration} iterations without a final_answer "
                "(max_iterations reached)."
            )
            route = ROUTE_DONE
        elif not state.research_notes:
            route = ROUTE_RESEARCHER
        elif not state.analysis_notes:
            route = ROUTE_ANALYST
        else:
            route = ROUTE_WRITER

        state.record_route(route)
        state.add_trace_event("supervisor", {"route": route, "iteration": state.iteration})

        return state
