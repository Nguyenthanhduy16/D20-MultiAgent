"""Search client abstraction for ResearcherAgent."""

import requests
from tavily import TavilyClient
from tavily.errors import TimeoutError as TavilyTimeoutError
from tavily.errors import UsageLimitExceededError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client, backed by Tavily."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.tavily_api_key:
            raise LabError("TAVILY_API_KEY is not set. Add it to your .env file.")

        self._timeout = float(settings.timeout_seconds)
        self._client = TavilyClient(api_key=settings.tavily_api_key)

    @retry(
        retry=retry_if_exception_type(
            (TavilyTimeoutError, UsageLimitExceededError, requests.exceptions.ConnectionError)
        ),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Retry (transient errors only) and timeout are handled here, not in agents.
        """

        response = self._client.search(query, max_results=max_results, timeout=self._timeout)

        return [
            SourceDocument(
                title=result.get("title") or result.get("url") or "Untitled",
                url=result.get("url"),
                snippet=result.get("content", ""),
                metadata={"score": result.get("score")},
            )
            for result in response.get("results", [])
        ]
