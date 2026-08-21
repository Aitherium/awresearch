"""Public API for awresearch — Researcher, Report, Claim, Source."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("awresearch")


@dataclass
class Source:
    """A source: URL, title, retrieved metadata."""

    url: str
    title: str
    retrieved_at: Optional[str] = None
    domain: Optional[str] = None
    authority: Optional[float] = None
    freshness: Optional[float] = None
    trust: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "url": self.url,
            "title": self.title,
            "retrieved_at": self.retrieved_at,
            "domain": self.domain,
            "authority": self.authority,
            "freshness": self.freshness,
            "trust": self.trust,
        }


@dataclass
class Claim:
    """A single claim in the report, with its sources."""

    text: str
    sources: list[int] = field(default_factory=list)  # 1-based indices into Report.sources
    unsourced_reason: Optional[str] = None

    @property
    def is_sourced(self) -> bool:
        """True if this claim has at least one source."""
        return bool(self.sources)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "text": self.text,
            "sources": self.sources,
            "is_sourced": self.is_sourced,
            "unsourced_reason": self.unsourced_reason,
        }


@dataclass
class Report:
    """A research report: claims, sources, and metadata."""

    question: str
    claims: list[Claim] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    raw_response: str = ""
    research_depth: str = "standard"

    def validate(self) -> list[str]:
        """Validate the report. Returns a list of issues found.

        Issues include:
        - Claims with no sources (unsourced claims without an explicit reason)
        - Sources referenced in claims that don't exist
        """
        issues = []
        for i, claim in enumerate(self.claims):
            if not claim.is_sourced and not claim.unsourced_reason:
                issues.append(
                    f"Claim {i}: '{claim.text[:50]}...' has no source and no reason"
                )
            for src_idx in claim.sources:
                if src_idx < 1 or src_idx > len(self.sources):
                    issues.append(
                        f"Claim {i} references source {src_idx}, but only {len(self.sources)} exist"
                    )
        return issues

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "question": self.question,
            "research_depth": self.research_depth,
            "claims": [c.to_dict() for c in self.claims],
            "sources": [s.to_dict() for s in self.sources],
            "raw_response": self.raw_response,
        }

    def markdown(self) -> str:
        """Render the report as Markdown with citations."""
        lines = [f"# {self.question}\n"]

        if not self.claims:
            lines.append("(No claims in this report)\n")
        else:
            for claim in self.claims:
                text = claim.text
                if claim.sources:
                    cite_nums = ", ".join(f"[{i}]" for i in claim.sources)
                    text = f"{text} {cite_nums}"
                elif claim.unsourced_reason:
                    text = f"{text} *(unsourced: {claim.unsourced_reason})*"
                lines.append(f"- {text}\n")

        if self.sources:
            lines.append("\n## Sources\n")
            for i, source in enumerate(self.sources, 1):
                lines.append(f"[{i}] {source.title}\n")
                lines.append(f"    {source.url}\n")
                if source.domain:
                    parts = [source.domain]
                    if source.authority is not None:
                        parts.append(f"authority: {source.authority:.2f}")
                    if source.freshness is not None:
                        parts.append(f"freshness: {source.freshness:.2f}")
                    lines.append(f"    ({', '.join(parts)})\n")
                lines.append("\n")

        return "".join(lines).rstrip() + "\n"


class Researcher:
    """Configurable research agent using pluggable search and completion backends.

    By default uses AitherSearch (multi-engine web search) and requires an awdk
    LLM backend. Backends are pluggable for testing and alternative implementations.

    Args:
        llm_backend: An awdk LLMRouter or compatible object (required)
        search_backend: A callable(query: str) -> list[dict] returning
            [{"title", "url", "snippet"}, ...]. Defaults to AitherSearch.
        artifacts_dir: Where to store temporary files. Defaults to a temp dir.
    """

    def __init__(
        self,
        llm_backend: Any,
        search_backend: Optional[Any] = None,
        artifacts_dir: Optional[Path] = None,
    ):
        self.llm_backend = llm_backend
        self.search_backend = search_backend or self._default_search_backend
        self.artifacts_dir = artifacts_dir or Path(os.getenv("AITHER_DATA_DIR", "/tmp"))
        self._session = None

    @staticmethod
    def _default_search_backend(query: str) -> list[dict]:
        """Default search: AitherSearch multi-engine."""
        try:
            from . import aithersearch

            result = asyncio.run(aithersearch.search_async(query, limit=10))
            return result.get("results", [])
        except Exception as exc:
            logger.warning("default search failed: %s", exc)
            return []

    async def research(
        self,
        question: str,
        depth: str = "standard",
        max_sources: int = 10,
    ) -> Report:
        """Research a question and return a cited report.

        Args:
            question: The research question
            depth: "standard" (one pass) or "deep" (iterative multi-angle)
            max_sources: Maximum sources to fetch

        Returns:
            A Report with claims, sources, and validation.

        The report enforces that every claim carries at least one source,
        or has an explicit unsourced_reason. A report is not considered
        complete until validated (validate() returns []).
        """
        raise NotImplementedError(
            "research() requires an AitherAgent integration; "
            "see awresearch README for full setup with awdk"
        )


__all__ = ["Researcher", "Report", "Claim", "Source"]
