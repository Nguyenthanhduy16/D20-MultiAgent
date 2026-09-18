# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this repo is

A **teaching skeleton** (Vietnamese-language lab, "Lab 20: Multi-Agent Research System") for a 2-hour multi-agent systems lab. It is intentionally incomplete: the architecture, schemas, config, and tests are production-grade and finished, but the core logic in each module is stubbed out with `raise StudentTodoError(...)`. Students fill in the TODOs; the repo is not meant to work out of the box.

Target architecture (from [README.md](README.md)):

```
User Query -> Supervisor/Router -> Researcher -> Analyst -> Writer -> Trace + Benchmark Report
```

When asked to "implement the lab" or "build the multi-agent system," treat the existing interfaces (`BaseAgent`, `ResearchState`, `Settings`, schemas) as fixed contracts to fill in, not things to redesign, unless the user explicitly asks for a redesign.

## Commands

```bash
pip install -e ".[dev,llm]"        # install (dev = pytest/ruff/mypy, llm = openai/langgraph/langsmith)
cp .env.example .env                # then fill in OPENAI_API_KEY etc.

make test        # pytest
make lint         # ruff check src tests
make format       # ruff format src tests
make typecheck    # mypy src (strict mode)
make run-baseline # python -m multi_agent_research_lab.cli baseline --query "..."
make run-multi    # python -m multi_agent_research_lab.cli multi-agent --query "..."

pytest tests/test_state.py -k test_state_records_route_and_trace   # single test
grep -R "TODO(student)" -n src tests docs   # find all unfinished lab work
```

CLI is also installed as a script entrypoint: `malab baseline --query "..."` / `malab multi-agent --query "..."`.

Before considering student work "done," it must pass `make lint`, `make typecheck`, and `make test` (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Architecture

### Data flow contract

Everything flows through a single mutable Pydantic model, `ResearchState` ([src/multi_agent_research_lab/core/state.py](src/multi_agent_research_lab/core/state.py)): each agent reads whatever fields it needs and writes its own output field, then returns the same state object. This is the one thing every other module depends on:

- `request: ResearchQuery` — the input
- `sources`, `research_notes`, `analysis_notes`, `final_answer` — worker outputs, populated in that order
- `route_history` / `iteration` — updated via `state.record_route(name)`, used for max-iteration guardrails
- `trace` — updated via `state.add_trace_event(name, payload)`
- `agent_results` — list of `AgentResult` (agent name + content + metadata), one per agent invocation
- `errors` — accumulated failure strings

### Agent contract

All agents in `src/multi_agent_research_lab/agents/` subclass `BaseAgent` ([agents/base.py](src/multi_agent_research_lab/agents/base.py)), which is a one-method interface: `run(state: ResearchState) -> ResearchState`. There are five: `supervisor`, `researcher`, `analyst`, `writer`, `critic` (critic is explicitly optional/bonus). The supervisor is not a special class — it's an agent like the others, whose job is to decide the next route and append to `route_history` rather than to produce content.

### Orchestration layer

`MultiAgentWorkflow` ([graph/workflow.py](src/multi_agent_research_lab/graph/workflow.py)) is the only place orchestration (LangGraph graph construction, node wiring, conditional routing, stop condition) should live. Agent internals must stay in `agents/`; keep that separation when implementing `build()`/`run()`. The CLI ([cli.py](src/multi_agent_research_lab/cli.py)) never talks to individual agents directly — only to `MultiAgentWorkflow` (multi-agent path) or `LLMClient` (baseline path).

### Service abstraction boundary

Agents must not call SDKs (OpenAI, Tavily, etc.) directly. They depend on `LLMClient.complete()` ([services/llm_client.py](src/multi_agent_research_lab/services/llm_client.py)) and `SearchClient.search()` ([services/search_client.py](src/multi_agent_research_lab/services/search_client.py)), both provider-agnostic skeletons. Retry/timeout/token-accounting logic belongs in these clients, not scattered across agents — `LLMResponse` already carries `input_tokens`/`output_tokens`/`cost_usd` for this reason.

### Config

`Settings` ([core/config.py](src/multi_agent_research_lab/core/config.py)) is a single `pydantic-settings` object loaded from `.env`, cached via `get_settings()`. Agents/services must go through `get_settings()`, not read `os.environ` directly — this is a stated convention, not just a style preference. Guardrails (`max_iterations`, `timeout_seconds`) live here and are meant to be enforced by the supervisor/workflow, not hardcoded per-agent. `configs/lab_default.yaml` holds per-agent model/temperature presets for the lab variant but is not currently wired into `Settings` — it's a reference config, not consumed by code yet.

### Errors

`core/errors.py` defines the exception vocabulary: `StudentTodoError` marks unfinished skeleton code (the CLI specifically catches this in the `multi-agent` command and prints it as an "Expected TODO" panel rather than a crash), `AgentExecutionError` is meant for retry/fallback-exhausted failures, `ValidationError` for state/output validation failures.

### Evaluation/observability (built after the workflow runs)

- `observability/tracing.py`: a provider-agnostic `trace_span` context manager; meant to be swapped for/augmented with LangSmith or Langfuse.
- `evaluation/benchmark.py`: `run_benchmark(run_name, query, runner)` times a runner and returns a `BenchmarkMetrics` skeleton (latency only; quality/cost/citation-coverage/failure-rate are TODOs).
- `evaluation/report.py`: `render_markdown_report` turns a list of `BenchmarkMetrics` into the markdown table for `reports/benchmark_report.md`.

## Working with the TODO(student) markers

- Every unfinished method raises `StudentTodoError` with a docstring above it describing the expected behavior — read the docstring before implementing, it encodes the intended design (e.g. supervisor's docstring lists the exact routing steps expected).
- [tests/test_agents_todo.py](tests/test_agents_todo.py) asserts that `SupervisorAgent.run` raises `StudentTodoError` — this is a placeholder "skeleton guard" test, not a real spec. It is *expected* to fail once the supervisor is implemented; replace it with a real routing test rather than trying to keep it green.
- Implementation order matters because of dependencies: `LLMClient` → `SearchClient` → CLI `baseline` command → worker agents (`researcher`/`analyst`/`writer`/optional `critic`) → `SupervisorAgent` → `MultiAgentWorkflow` → tracing/benchmark/report. Don't start the graph/supervisor before the worker agents and services exist, since they have nothing to call.
- `docs/lab_guide.md` and `docs/codelab.md` contain the full milestone-by-milestone instructions and a macOS SSL troubleshooting note for search/LLM HTTPS calls; `docs/design_template.md` is the design doc students fill in before coding; `docs/peer_review_rubric.md` is the grading rubric.

## Foreign agent config detected

An OpenAI Codex config was found at `~/.codex/config.toml` (user-level, outside this repo). Reply `/import` to scan it and list what's importable (MCP servers, slash commands, subagents, skills, instructions), then `/import --yes=<digest>` to apply.
