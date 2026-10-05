"""End-to-end tests for Researcher.research() and the awresearch CLI.

Everything external is faked: the LLM (routes on the system prompt of each
pipeline role), web search, page fetching (an httpx MockTransport) and the
knowledge graph. The real pipeline in deep_mode/tools runs unmodified.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("adk")

from awresearch import api, cli, tools  # noqa: E402
from awresearch.api import LLMUnavailableError, Researcher  # noqa: E402
from awresearch.ledger import SavingsLedger  # noqa: E402

URL_A = "https://alpha.example.org/webgpu"
URL_B = "https://beta.example.com/browsers"
INVENTED = "https://invented.example.net/not-read"

PAGES = {
    URL_A: "<html><body><article><h1>WebGPU</h1>"
           "<p>WebGPU is a web API for GPU graphics and compute, the successor to WebGL. "
           "It exposes modern GPU features to web pages.</p>" * 3 + "</article></body></html>",
    URL_B: "<html><body><article><h1>Browser support</h1>"
           "<p>Chrome shipped WebGPU in version 113. Safari and Firefox followed later "
           "on some platforms.</p>" * 3 + "</article></body></html>",
}


def _resp(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=text, prompt_tokens=10, completion_tokens=5, cache_status="")


class FakeLLM:
    """Answers each pipeline role by recognising its system prompt."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _answer(self, messages) -> str:
        system = messages[0].content
        user = messages[-1].content
        self.calls.append(system[:40])
        if "single word OK" in system:
            return "OK"
        if "research director" in system:
            return json.dumps(["What is WebGPU?", "Which browsers ship WebGPU?"])
        if "Extract only facts" in system:
            out = []
            if URL_A in user:
                out.append({"claim": "WebGPU is a web API for GPU graphics and compute.",
                            "source_url": URL_A})
                # A URL the extractor made up: must NOT become a citation.
                out.append({"claim": "WebGPU is used by most games.",
                            "source_url": INVENTED})
            if URL_B in user:
                out.append({"claim": "Chrome shipped WebGPU in version 113.",
                            "source_url": URL_B})
            return "```json\n" + json.dumps(out) + "\n```"
        if "adversarial fact-checker" in system:
            n = user.count("CLAIM:")
            return json.dumps([{"index": i, "supported": True, "reason": "stated"}
                               for i in range(n)])
        if "Write a precise" in system:
            nums = [ln.split("]")[0] + "]" for ln in user.splitlines() if ln.startswith("[")]
            return ("<think>\nreasoning [9] that must not reach the report\n</think>\n"
                    "WebGPU is a GPU API " + " ".join(nums) + ".")
        return ""

    async def chat(self, messages, **kwargs):
        return _resp(self._answer(messages))

    async def chat_stream(self, messages, **kwargs):
        yield _resp(self._answer(messages))


class DeadLLM:
    """A backend that is configured but never answers."""

    async def chat(self, messages, **kwargs):
        raise ConnectionError("connection refused\nProvider response: {\"error\": \"down\"}")

    async def chat_stream(self, messages, **kwargs):
        raise ConnectionError("connection refused")
        yield  # pragma: no cover


class FakeGraph:
    async def search(self, query, limit=5):
        return []

    async def add_node(self, **kwargs):
        return SimpleNamespace(id=1, content=kwargs.get("content", ""), label="x")

    async def add_edge(self, *args, **kwargs):
        return None


def fake_search(query: str) -> list[dict]:
    return [
        {"title": "WebGPU explained", "url": URL_A, "snippet": "GPU API"},
        {"title": "Browser support for WebGPU", "url": URL_B, "snippet": "Chrome 113"},
    ]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Keep adk state in tmp and serve page fetches from PAGES."""
    monkeypatch.setenv("AITHER_DATA_DIR", str(tmp_path / "adk"))
    monkeypatch.setenv("AITHER_GRAPH_AUTOSYNC", "false")

    def handler(request: httpx.Request) -> httpx.Response:
        body = PAGES.get(str(request.url))
        if body is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})

    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(tools.httpx, "AsyncClient", client_factory)


def _researcher(llm=None) -> Researcher:
    return Researcher(llm_backend=llm or FakeLLM(), search_backend=fake_search,
                      graph=FakeGraph())


# ── Researcher.research() ────────────────────────────────────────────────────
@pytest.mark.parametrize("depth", ["standard", "deep"])
async def test_claims_cite_real_source_indices(depth):
    events: list[dict] = []
    report = await _researcher().research("What is WebGPU?", depth=depth,
                                          on_event=events.append)

    assert report.research_depth == depth
    assert report.validate() == []
    urls = [s.url for s in report.sources]
    assert set(urls) == {URL_A, URL_B}
    assert INVENTED not in urls
    for s in report.sources:
        assert s.domain and s.title and s.retrieved_at

    sourced = [c for c in report.claims if c.is_sourced]
    assert len(sourced) == 2
    for c in sourced:
        for idx in c.sources:
            src = report.sources[idx - 1]  # 1-based
            word = "Chrome" if "Chrome" in c.text else "GPU graphics"
            assert word in PAGES[src.url]
    # The synthesized answer's citations point into Report.sources.
    assert "[1]" in report.raw_response and "[2]" in report.raw_response
    assert report.raw_response.startswith("WebGPU is a GPU API")  # <think> stripped
    assert events and all(set(e) == {"phase", "message"} for e in events)
    assert events[-1]["phase"] == "report"
    if depth == "deep":
        assert {"decompose", "verify"} <= {e["phase"] for e in events}


async def test_unsourced_claim_carries_reason():
    report = await _researcher().research("What is WebGPU?")
    unsourced = [c for c in report.claims if not c.is_sourced]
    assert len(unsourced) == 1
    assert unsourced[0].text == "WebGPU is used by most games."
    assert INVENTED in unsourced[0].unsourced_reason
    assert "not one of the pages read" in unsourced[0].unsourced_reason
    md = report.markdown()
    assert "unsourced:" in md and "## Sources" in md


async def test_dead_backend_raises_llm_unavailable():
    with pytest.raises(LLMUnavailableError, match="did not answer"):
        await _researcher(DeadLLM()).research("What is WebGPU?")


async def test_refused_default_model_falls_back_to_a_listed_model(monkeypatch):
    class Provider:
        default_model = "big"

        async def list_models(self):
            return ["big", "small"]

    class Router(FakeLLM):
        provider_name = "fake"

        def __init__(self, model):
            super().__init__()
            self.model = model

        async def get_provider(self):
            return Provider()

        async def chat(self, messages, **kwargs):
            if self.model is None:
                raise RuntimeError("403 Forbidden\nProvider response: {\"error\": "
                                   "\"Model 'big' requires starter tier\"}")
            return await super().chat(messages, **kwargs)

    monkeypatch.setattr(api, "_default_llm", lambda model, ledger: Router(model))
    r = Researcher(search_backend=fake_search, graph=FakeGraph())
    report = await r.research("What is WebGPU?")
    assert r.model_used == "small"
    assert report.claims


def test_default_llm_honours_adk_config(monkeypatch):
    """The default router carries adk's Config, so its documented knobs
    (AITHER_LLM_BACKEND, `adk setup`) apply exactly as for an AitherAgent."""
    monkeypatch.setenv("AITHER_LLM_BACKEND", "vllm")
    llm = api._default_llm("m1", ledger=SavingsLedger())
    assert llm._config is not None and llm._config.llm_backend == "vllm"


# ── CLI ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def cli_fakes(monkeypatch):
    """Route the CLI's default Researcher onto the fakes."""
    monkeypatch.setattr(api, "_default_llm", lambda model, ledger: FakeLLM())
    monkeypatch.setattr(api.Researcher, "_build_graph", lambda self, d: FakeGraph())

    async def search(query, limit=6):
        return {"query": query, "results": fake_search(query)}

    monkeypatch.setattr(tools.aithersearch, "search", search)


def test_cli_writes_valid_json_to_out_file(cli_fakes, tmp_path, capsys):
    out = tmp_path / "report.json"
    rc = cli.main(["--question", "What is WebGPU?", "--output", "json",
                   "--out-file", str(out)])
    assert rc == 0
    assert capsys.readouterr().out == ""
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["question"] == "What is WebGPU?"
    assert data["research_depth"] == "standard"
    n = len(data["sources"])
    assert n == 2
    for c in data["claims"]:
        assert c["sources"] or c["unsourced_reason"]
        assert all(1 <= i <= n for i in c["sources"])


def test_cli_events_are_json_lines_on_stderr(cli_fakes, capsys):
    rc = cli.main(["--question", "What is WebGPU?", "--depth", "deep", "--events"])
    assert rc == 0
    cap = capsys.readouterr()
    assert cap.out.startswith("# What is WebGPU?")
    events = [json.loads(ln) for ln in cap.err.splitlines() if ln.startswith("{")]
    assert len(events) >= 5
    assert all(set(e) == {"phase", "message"} for e in events)
    phases = [e["phase"] for e in events]
    assert phases[0] == "connect" and phases[-1] == "report"
    assert "search" in phases and "synthesize" in phases


def test_cli_without_events_keeps_stderr_quiet(cli_fakes, capsys):
    assert cli.main(["--question", "What is WebGPU?"]) == 0
    assert capsys.readouterr().err == ""


def test_cli_no_llm_exits_1_with_one_line_reason(monkeypatch, capsys):
    class NoBackend:
        provider_name = ""

        async def get_provider(self):
            raise ConnectionError("No LLM backend available.\n\n  Run setup: ...")

    monkeypatch.setattr(api, "_default_llm", lambda model, ledger: NoBackend())
    rc = cli.main(["--question", "What is WebGPU?", "--output", "json"])
    assert rc == 1
    cap = capsys.readouterr()
    assert cap.out == ""
    lines = [ln for ln in cap.err.splitlines() if ln.strip()]
    assert lines == ["awresearch: no LLM backend reachable: No LLM backend available."]
    assert "Traceback" not in cap.err


def test_cli_requires_question(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
