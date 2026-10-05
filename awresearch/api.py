"""Public API for awresearch — Researcher, Report, Claim, Source."""

from __future__ import annotations

import inspect
import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Union
from urllib.parse import urlparse

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

        if self.raw_response.strip():
            lines.append("\n## Summary\n\n")
            lines.append(self.raw_response.strip() + "\n")
            lines.append("\n## Claims\n\n")

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


class LLMUnavailableError(RuntimeError):
    """No LLM backend answered. ``str(exc)`` is a one-line, human-readable reason."""


#: A progress callback: receives ``{"phase": str, "message": str}`` dicts.
EventCallback = Callable[[dict], Union[None, Awaitable[None]]]

#: Errors that mean "this backend is up but will not serve THIS model to you"
#: (tier gate, unknown model, auth) -- worth trying another listed model.
_REFUSED = re.compile(
    r"\b(?:401|402|403|404)\b|requires\s+\S+\s+tier|unknown model|model .*not found",
    re.IGNORECASE,
)

_DEPTHS = ("standard", "deep")
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _one_line(exc: BaseException, limit: int = 300) -> str:
    """Collapse an exception to one readable line: its first line plus, when
    present, the provider's own error line (which usually names the cause)."""
    text = str(exc).strip() or type(exc).__name__
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = lines[0] if lines else type(exc).__name__
    detail = next((ln for ln in lines[1:] if "error" in ln.lower()), "")
    if detail and detail not in out:
        out = f"{out} -- {detail}"
    return out[:limit]


def _norm_url(url: str) -> str:
    """Key for matching a URL across the page cache, the citation list and findings."""
    return (url or "").strip().split("#", 1)[0].rstrip("/")


def _as_float(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _default_llm(model: Optional[str], ledger: Any) -> Any:
    """adk's LLMRouter with default auto-detection (metered into ``ledger``).

    No provider, URL or key is chosen here: adk resolves the backend from its own
    configuration (``adk setup``, env keys, a local runtime). Module-level so tests
    can substitute it.
    """
    from .ledger import LedgerRouter

    if model:
        return LedgerRouter(model=model, ledger=ledger)
    return LedgerRouter(ledger=ledger)


async def _ping(llm: Any) -> Optional[str]:
    """One tiny completion. Returns None when the backend answered, else the reason."""
    from adk.llm.base import Message

    try:
        await llm.chat(
            [Message(role="system", content="Reply with the single word OK."),
             Message(role="user", content="Are you there?")],
            effort=2,
        )
        return None
    except Exception as exc:  # noqa: BLE001 -- any failure is "did not answer"
        return _one_line(exc)


class _MeteredLLM:
    """Wraps the agent's LLM to count calls that actually answered.

    The research pipeline deliberately swallows LLM errors so a single bad call
    degrades instead of crashing; without this counter a backend that refused
    every call would produce an empty report that looks like "nothing found".
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self.answered = 0
        self.failed = 0
        self.last_error = ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def chat(self, messages: Any, **kwargs: Any) -> Any:
        try:
            resp = await self._inner.chat(messages, **kwargs)
        except Exception as exc:
            self.failed += 1
            self.last_error = _one_line(exc)
            raise
        self.answered += 1
        return resp

    async def chat_stream(self, messages: Any, **kwargs: Any) -> Any:
        got = False
        try:
            async for chunk in self._inner.chat_stream(messages, **kwargs):
                got = True
                yield chunk
        except Exception as exc:
            self.failed += 1
            self.last_error = _one_line(exc)
            raise
        if got:
            self.answered += 1


def _to_progress(ev: dict) -> Optional[dict]:
    """Translate a pipeline event into ``{"phase", "message"}``; None = not shown."""
    kind = ev.get("type")
    if kind == "stage":
        stage = str(ev.get("stage", ""))
        start = ev.get("status") == "start"
        if stage == "decompose":
            if start:
                return {"phase": stage, "message": "breaking the question into sub-questions"}
            subqs = ev.get("subquestions") or []
            return {"phase": stage, "message": f"{len(subqs)} sub-questions: " + " | ".join(subqs)}
        if stage == "research":
            pos = f"[{int(ev.get('index', 0)) + 1}/{ev.get('total', 1)}]"
            if start:
                return {"phase": stage, "message": f"{pos} researching: {ev.get('subq', '')}"}
            return {"phase": stage, "message": f"{pos} {ev.get('found', 0)} findings"}
        if stage == "verify":
            if start:
                return {"phase": stage,
                        "message": f"checking {ev.get('count', 0)} findings against their sources"}
            return {"phase": stage, "message": f"{ev.get('supported', 0)} supported, "
                                               f"{ev.get('dropped', 0)} dropped"}
        if stage == "synthesize":
            return {"phase": stage,
                    "message": "writing the cited answer" if start else "answer written"}
        return {"phase": stage or "stage", "message": str(ev.get("status", ""))}
    if kind == "tool":
        name = ev.get("name")
        args = ev.get("args") or {}
        if name == "web_search":
            return {"phase": "search", "message": str(args.get("query", ""))}
        if name == "fetch_many":
            urls = args.get("urls") or []
            return {"phase": "read", "message": f"reading {len(urls)} pages"}
        return None
    return None


class Researcher:
    """Research agent: search, read the real pages, extract claims with citations.

    With no arguments it builds everything it needs: adk's default LLM routing
    (whatever ``adk`` resolves on this machine), keyless multi-engine web search,
    and a throwaway knowledge graph.

    Args:
        llm_backend: An adk ``LLMRouter`` or any object with async ``chat`` and
            ``chat_stream``. None = adk's default routing, probed before use.
        search_backend: Optional ``callable(query) -> list[{"title","url","snippet"}]``
            (sync or async) replacing the built-in web search.
        artifacts_dir: Where the session's graph/memory live. None = a temporary
            directory removed after the run (each report starts from a clean slate).
        model: Model name to request. None = the backend's default; if the backend
            refuses its default model, the other models it lists are tried in order.
        graph: Optional knowledge-graph object (``async search``/``add_node``/
            ``add_edge``). None = an adk ``GraphMemory`` under ``artifacts_dir``.
    """

    def __init__(
        self,
        llm_backend: Any = None,
        search_backend: Optional[Callable[[str], Any]] = None,
        artifacts_dir: Optional[Path] = None,
        model: Optional[str] = None,
        graph: Any = None,
    ):
        self.llm_backend = llm_backend
        self.search_backend = search_backend
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.model = model
        self.graph = graph
        #: The model the last run used, when known.
        self.model_used: Optional[str] = None
        #: The adk SavingsLedger snapshot of the last run (tokens, searches, pages).
        self.last_usage: dict[str, Any] = {}

    # ── LLM ──────────────────────────────────────────────────────────────────
    async def _connect_llm(self, ledger: Any, emit: Callable[[str, str], Awaitable[None]]) -> Any:
        """Return an LLM that has answered a ping, or raise LLMUnavailableError."""
        if self.llm_backend is not None:
            err = await _ping(self.llm_backend)
            if err:
                raise LLMUnavailableError(f"the LLM backend did not answer: {err}")
            self.model_used = self.model
            return self.llm_backend

        await emit("connect", "finding an LLM backend")
        llm = _default_llm(self.model, ledger)
        provider = None
        if hasattr(llm, "get_provider"):
            try:
                provider = await llm.get_provider()
            except Exception as exc:  # noqa: BLE001
                raise LLMUnavailableError(
                    f"no LLM backend reachable: {_one_line(exc)}") from exc
        backend = getattr(llm, "provider_name", "") or "default"
        current = getattr(provider, "default_model", "") or self.model or ""
        err = await _ping(llm)
        if err is None:
            self.model_used = current or None
            await emit("connect", f"using {backend} backend, model {current or '(default)'}")
            return llm
        if self.model or not _REFUSED.search(err) or provider is None:
            raise LLMUnavailableError(f"the {backend} LLM backend did not answer: {err}")

        # The backend is up but refused its default model (tier/auth/unknown):
        # try the other models it advertises before giving up.
        try:
            models = [m for m in (await provider.list_models() or []) if m and m != current]
        except Exception:  # noqa: BLE001
            models = []
        await emit("connect", f"model {current or '(default)'} was refused; "
                              f"trying the {len(models)} other model(s) the backend lists")
        tried: list[str] = []
        for name in models:
            cand = _default_llm(name, ledger)
            cerr = await _ping(cand)
            if cerr is None:
                self.model_used = name
                await emit("connect", f"using {backend} backend, model {name}")
                return cand
            tried.append(f"{name}: " + ("refused" if _REFUSED.search(cerr) else cerr[:120]))
        raise LLMUnavailableError(
            (f"the {backend} LLM backend refused model {current or '(default)'} ({err})"
             + ("; other models -- " + "; ".join(tried) if tried else ""))[:600])

    # ── agent + session ──────────────────────────────────────────────────────
    def _build_agent(self, llm: Any, data_dir: Path) -> Any:
        """An AitherAgent with ONLY the curated research tools registered."""
        from adk.agent import AitherAgent
        from adk.identity import Identity
        from adk.memory import Memory

        identity = Identity(
            name="researcher",
            description="Research analyst: searches, reads sources, cites every claim.",
            skills=["deep_research", "fact_checking"],
        )
        agent = AitherAgent(
            name="researcher",
            identity=identity,
            llm=llm,
            memory=Memory(db_path=str(data_dir / "memory.db"), agent_name="researcher"),
            builtin_tools=False,
            user_mcp=False,
            phonehome=False,
        )
        return agent

    def _build_graph(self, data_dir: Path) -> Any:
        if self.graph is not None:
            return self.graph
        from adk.graph_memory import GraphMemory

        return GraphMemory(db_path=str(data_dir / "graph.db"), agent_name="researcher",
                           fleet_url="", auto_sync=False)

    def _search_tool(self, session: Any) -> Callable[..., Awaitable[str]]:
        """A ``web_search`` tool backed by the caller's ``search_backend``."""
        backend = self.search_backend

        async def web_search(query: str, limit: int = 6) -> str:
            """Search the web. Returns titles, URLs, and snippets.

            query: what to search for
            limit: max results (default 6)
            """
            session.ledger.note_search()
            try:
                res = backend(str(query))
                if inspect.isawaitable(res):
                    res = await res
            except Exception as exc:  # noqa: BLE001
                logger.warning("search backend failed: %s", exc)
                res = []
            if isinstance(res, dict):
                res = res.get("results", [])
            results = [r for r in (res or []) if isinstance(r, dict)][: int(limit or 6)]
            for r in results:
                if r.get("url"):
                    session.cite(r.get("title", ""), r["url"])
            return json.dumps({"query": query, "results": results, "count": len(results)})

        return web_search

    # ── the run ──────────────────────────────────────────────────────────────
    async def research(
        self,
        question: str,
        depth: str = "standard",
        max_sources: int = 10,
        on_event: Optional[EventCallback] = None,
    ) -> Report:
        """Research a question and return a cited report.

        Args:
            question: The research question.
            depth: ``"standard"`` (one search-read-extract pass over the question)
                or ``"deep"`` (decompose into sub-questions, research them in
                parallel, adversarially verify every claim, then synthesize).
            max_sources: Page-read budget for the run.
            on_event: Optional progress callback receiving
                ``{"phase": str, "message": str}`` (sync or async).

        Returns:
            A Report. Every claim either cites 1-based indices into
            ``Report.sources`` (pages that were actually read) or carries an
            ``unsourced_reason``; ``raw_response`` is the synthesized answer with
            its citations renumbered to match ``Report.sources``.

        Raises:
            ValueError: ``question`` is empty or ``depth`` is unknown.
            LLMUnavailableError: no LLM backend answered (before or during the run).
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("question is empty")
        if depth not in _DEPTHS:
            raise ValueError(f"depth must be one of {_DEPTHS}, got {depth!r}")
        max_sources = max(1, int(max_sources))

        async def emit(phase: str, message: str) -> None:
            if on_event is None:
                return
            try:
                r = on_event({"phase": phase, "message": message})
                if inspect.isawaitable(r):
                    await r
            except Exception as exc:  # noqa: BLE001 -- progress must never break a run
                logger.debug("progress callback failed: %s", exc)

        async def pipeline_event(ev: dict) -> None:
            p = _to_progress(ev)
            if p:
                await emit(p["phase"], p["message"])

        from .deep_mode import run_deep_research, run_standard_research
        from .ledger import SavingsLedger
        from .tools import ResearchSession, build_research_tools

        own_dir = self.artifacts_dir is None
        data_dir = Path(tempfile.mkdtemp(prefix="awresearch-")) if own_dir else self.artifacts_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            ledger = SavingsLedger()
            llm = _MeteredLLM(await self._connect_llm(ledger, emit))
            agent = self._build_agent(llm._inner, data_dir)
            agent.llm = llm
            session = ResearchSession(graph=self._build_graph(data_dir), ledger=ledger,
                                      artifacts_dir=data_dir, session_id="awresearch")
            for fn in build_research_tools(session):
                agent._tools.register(fn)
            if self.search_backend is not None:
                agent._tools.register(self._search_tool(session))

            if depth == "deep":
                answer, _ = await run_deep_research(agent, session, question, pipeline_event,
                                                    max_sources=max_sources)
            else:
                answer, _ = await run_standard_research(agent, session, question,
                                                        pipeline_event, max_sources=max_sources)

            if llm.answered == 0:
                raise LLMUnavailableError(
                    "the LLM backend answered the probe but no research call: "
                    + (llm.last_error or "every call returned nothing"))
            self.last_usage = ledger.snapshot()
            report = _build_report(question, depth, session, answer or "")
            await emit("report", f"{len(report.claims)} claims, {len(report.sources)} sources")
            return report
        finally:
            if own_dir:
                shutil.rmtree(data_dir, ignore_errors=True)


def _build_report(question: str, depth: str, session: Any, answer: str) -> Report:
    """Map the session's findings + citation list onto a Report.

    A finding cites a source only if its URL is a page that was actually read
    this session; anything else stays in the report as an unsourced claim with
    the reason, so a model-invented URL can never pass as a citation.
    """
    read = {_norm_url(u) for u in getattr(session, "_page_cache", {})}
    meta = {_norm_url(s.get("url", "")): s for s in getattr(session, "sources", [])}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = Report(question=question, research_depth=depth)
    index_of: dict[str, int] = {}
    by_text: dict[str, Claim] = {}

    for f in getattr(session, "findings", None) or []:
        text = str(f.get("claim", "")).strip()
        url = str(f.get("source_url", "")).strip()
        if not text:
            continue
        key = re.sub(r"\s+", " ", text.lower())
        nu = _norm_url(url)
        if url and nu in read:
            if nu not in index_of:
                m = meta.get(nu, {})
                src_url = m.get("url") or url
                report.sources.append(Source(
                    url=src_url,
                    title=(m.get("title") or "").strip() or src_url,
                    retrieved_at=now,
                    domain=m.get("domain") or urlparse(src_url).netloc or None,
                    authority=_as_float(m.get("authority")),
                    freshness=_as_float(m.get("freshness")),
                    trust=_as_float(m.get("trust")),
                ))
                index_of[nu] = len(report.sources)
            idx = index_of[nu]
            claim = by_text.get(key)
            if claim is None:
                claim = Claim(text=text, sources=[idx])
                by_text[key] = claim
                report.claims.append(claim)
            elif idx not in claim.sources:
                claim.sources.append(idx)
                claim.unsourced_reason = None
        elif key not in by_text:
            reason = ("the model gave no source URL for this claim" if not url else
                      f"the cited URL {url} is not one of the pages read in this session")
            claim = Claim(text=text, sources=[], unsourced_reason=reason)
            by_text[key] = claim
            report.claims.append(claim)

    report.raw_response = _renumber(_strip_reasoning(answer), session, index_of)
    return report


def _strip_reasoning(text: str) -> str:
    """Drop a reasoning model's ``<think>...</think>`` preamble from the answer."""
    text = text or ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return text.replace("<think>", "").strip()


def _renumber(answer: str, session: Any, index_of: dict[str, int]) -> str:
    """Rewrite the answer's session citation numbers to Report.sources numbers;
    a citation with no matching report source is removed rather than left dangling."""
    sources = getattr(session, "sources", [])

    def sub(m: "re.Match[str]") -> str:
        out: list[int] = []
        for part in m.group(1).split(","):
            n = int(part.strip())
            if 1 <= n <= len(sources):
                k = index_of.get(_norm_url(sources[n - 1].get("url", "")))
                if k and k not in out:
                    out.append(k)
        return "".join(f"[{k}]" for k in out)

    return _CITE.sub(sub, answer or "").strip()


__all__ = ["Researcher", "Report", "Claim", "Source", "LLMUnavailableError"]
