"""Unit tests for SupervisorAgent's routing policy."""

from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_routes_to_researcher_when_no_notes() -> None:
    state = SupervisorAgent().run(_state())

    assert state.route_history[-1] == ROUTE_RESEARCHER
    assert state.iteration == 1


def test_routes_to_analyst_when_research_notes_present() -> None:
    state = _state()
    state.research_notes = "some notes"

    state = SupervisorAgent().run(state)

    assert state.route_history[-1] == ROUTE_ANALYST


def test_routes_to_writer_when_analysis_notes_present() -> None:
    state = _state()
    state.research_notes = "some notes"
    state.analysis_notes = "some analysis"

    state = SupervisorAgent().run(state)

    assert state.route_history[-1] == ROUTE_WRITER


def test_routes_to_done_when_final_answer_present() -> None:
    state = _state()
    state.final_answer = "the answer"

    state = SupervisorAgent().run(state)

    assert state.route_history[-1] == ROUTE_DONE


def test_stops_at_max_iterations_without_final_answer() -> None:
    state = _state()
    state.iteration = get_settings().max_iterations

    state = SupervisorAgent().run(state)

    assert state.route_history[-1] == ROUTE_DONE
    assert state.errors
