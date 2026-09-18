"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.utils.timer import elapsed_timer
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline placeholder."""

    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)
    system_prompt = (
        "You are a helpful research assistant. "
        "Answer accurately, clearly, and concisely. "
        "If you are uncertain, say so."
    )
    user_prompt = (
        f"Research question: {request.query}\n"
        f"Audience: {request.audience}\n\n"
        "Provide a useful answer to this question."
    )
    with elapsed_timer() as elapsed:
        response = LLMClient().complete(system_prompt, user_prompt)
    state.final_answer = response.content

    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_seconds": elapsed(),
            },
        )
    )
    user_prompt = (
        f"Research question: {request.query}\n"
        f"Audience: {request.audience}\n\n"
        "Provide a useful answer to this question."
    )

    state.add_trace_event(
        "baseline",
        {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_seconds": elapsed(),
        },
    )

    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
