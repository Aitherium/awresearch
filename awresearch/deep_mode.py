"""Tier 2 — Multi-Agent Deep Mode.

Server-side orchestration LAYERED OVER the existing AitherAgent + curated tools.
Instead of the single ReAct loop (serve.research_turn -> agent.stream_chat), deep
mode runs a fixed four-role pipeline:

    Director (decompose)  ->  parallel Sub-researchers (search + read + extract)
                          ->  adversarial Verifier (refute every claim)
                          ->  Synthesizer (one cited answer, streamed)

Design rules honoured here:
  - All retrieval goes through the REGISTERED tools by name via
    ``agent._tools.execute(name, args)`` — the genuine web_search/fetch_many/
    save_finding run, so the SavingsLedger counters (note_search/note_page_read/
    record_dedup/note_finding) and the citable ``session.sources`` are credited
    truthfully. We never fabricate work.
  - Every LLM call goes through ``agent.llm`` (the LedgerRouter) so all four roles
    are metered automatically (record_llm / record_stream).
  - DeepSeek-safe BY CONSTRUCTION: each LLM call is a freshly built
    ``[Message(system), Message(user)]`` list — leading system, single user, ZERO
    tool-role messages, no native assistant tool_calls to pair. The pipeline calls
    tools itself rather than through the model's native tool-calling, so the
    strict-template invariant the msg-structure test guards can't be violated.
  - Never raises: every model output is parsed loosely and degrades to a sane
    fallback (subqs -> [question], findings -> [], verify -> keep+flag).

``run_deep_research(agent, session, question, on_event)`` emits ``stage`` events
alongside the existing ``tool``/``tool_result``/``token`` events; serve.gen()
forwards arbitrary event types generically, and the UI lights up the pipeline
strip from the ``stage`` events.
"""

from __future__ import annotations

import asyncio
import json
import logging

from adk.llm.base import Message

logger = logging.getLogger("deep_research.deep_mode")

# Pipeline bounds (keep cost predictable — see the cost note in the design spec).
_MIN_SUBQ = 3
_MAX_SUBQ = 5
_MAX_URLS_PER_SUBQ = 4
_FANOUT = 3                # concurrent sub-researchers
_PAGE_CHARS = 4000        # chars of each page fed to the extractor
_VERIFY_EXCERPT = 1200    # chars of source text shown to the verifier per claim


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
async def _emit(on_event, ev: dict) -> None:
    """Forward an SSE-shaped event, mirroring stream_chat._emit: never raise, and
    await the callback if it returns a coroutine."""
    if not on_event:
        return
    try:
        r = on_event(ev)
        if asyncio.iscoroutine(r):
            await r
    except Exception as exc:  # noqa: BLE001 — emitting must never break the turn
        logger.debug("deep_mode emit failed: %s", exc)


async def _llm(agent, system: str, user: str, effort: int = 2) -> str:
    """One metered, DeepSeek-safe LLM call: EXACTLY two messages — a leading
    ``system`` then a single ``user`` — no tool roles, no interleave. Auto-metered
    via LedgerRouter.record_llm. Returns the stripped text (''/empty on failure)."""
    try:
        resp = await agent.llm.chat(
            [Message(role="system", content=system),
             Message(role="user", content=user)],
            effort=effort,
        )
        return (getattr(resp, "content", "") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("deep_mode llm call failed: %s", exc)
        return ""


def _loads_loose(text: str):
    """Best-effort JSON parse of a model reply. Strips ```json fences and slices
    from the first bracket to the matching last bracket. Returns {} or [] on miss,
    never raises."""
    if not text:
        return {}
    s = text.strip()
    if s.startswith("```"):
        # drop a leading ```json / ``` fence and a trailing ```
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    # find the outermost JSON value
    starts = [i for i in (s.find("["), s.find("{")) if i != -1]
    if not starts:
        return {}
    start = min(starts)
    open_ch = s[start]
    close_ch = "]" if open_ch == "[" else "}"
    end = s.rfind(close_ch)
    if end == -1 or end < start:
        return {} if open_ch == "{" else []
    frag = s[start:end + 1]
    try:
        return json.loads(frag)
    except Exception:  # noqa: BLE001
        try:
            return json.loads(s)
        except Exception:  # noqa: BLE001
            return [] if open_ch == "[" else {}


async def _exec(agent, on_event, name: str, args: dict) -> str:
    """Run a REGISTERED tool by name, emitting the same tool/tool_result events the
    UI + serve.py ledger-snapshot path already understand. Returns the raw string
    result. Genuine tool => honest ledger credit + real citable sources."""
    await _emit(on_event, {"type": "tool", "name": name, "args": args})
    try:
        r = await agent._tools.execute(name, args)
    except Exception as exc:  # noqa: BLE001
        r = json.dumps({"error": str(exc)})
    result = str(r)
    await _emit(on_event, {"type": "tool_result", "name": name, "result": result[:1500]})
    return result


def _loads_tool(result: str):
    """Parse a tool's JSON string result; tolerate non-JSON / errors."""
    try:
        obj = json.loads(result)
        return obj if isinstance(obj, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — one sub-researcher (search -> read -> focused extraction)
# ─────────────────────────────────────────────────────────────────────────────
async def _research_subq(agent, session, subq: str, i: int, n: int,
                         sem: asyncio.Semaphore, on_event) -> list[dict]:
    """Search the sub-question, read the top pages, and extract ONLY source-backed
    claims. Persists each finding through the real save_finding tool (dedup + graph
    + stable citation). Returns the findings tagged with their sub-question."""
    async with sem:
        await _emit(on_event, {"type": "stage", "stage": "research", "status": "start",
                               "subq": subq, "index": i, "total": n})

        search = _loads_tool(await _exec(agent, on_event, "web_search",
                                         {"query": subq, "limit": 6}))
        urls = [r["url"] for r in search.get("results", []) if r.get("url")][:_MAX_URLS_PER_SUBQ]

        pages: list[dict] = []
        if urls:
            fetched = _loads_tool(await _exec(agent, on_event, "fetch_many",
                                              {"urls": urls, "max_chars": _PAGE_CHARS}))
            pages = fetched.get("pages", []) or []

        # Build a numbered, URL-labelled corpus from the readable pages.
        corpus_blocks: list[str] = []
        for p in pages:
            txt = (p.get("text") or "").strip()
            if not txt or p.get("error"):
                continue
            corpus_blocks.append(f"SOURCE_URL: {p.get('url', '')}\n{txt}")

        findings: list[dict] = []
        if corpus_blocks:
            corpus = "\n\n---\n\n".join(corpus_blocks)[:14000]
            system = (
                "Extract only facts that DIRECTLY answer the sub-question and are "
                "explicitly supported by the provided sources. Return ONLY JSON: "
                '[{"claim": "...", "source_url": "..."}]. No claim without a '
                "source_url drawn from the SOURCE_URL lines below. No outside "
                "knowledge, no speculation."
            )
            user = (f"Sub-question: {subq}\n\nSources:\n\n{corpus}\n\n"
                    "Return the JSON array now.")
            raw = await _llm(agent, system, user, effort=2)
            parsed = _loads_loose(raw)
            if isinstance(parsed, list):
                for f in parsed:
                    if not isinstance(f, dict):
                        continue
                    claim = str(f.get("claim", "")).strip()
                    src = str(f.get("source_url", "")).strip()
                    if claim and src:
                        findings.append({"claim": claim, "source_url": src, "subq": subq})

        # Persist via the genuine tool — dedups, writes the graph, credits
        # note_finding, and registers a STABLE citation index for source_url.
        for f in findings:
            await _exec(agent, on_event, "save_finding",
                        {"claim": f["claim"], "source_url": f["source_url"], "topic": subq})

        await _emit(on_event, {"type": "stage", "stage": "research", "status": "done",
                               "subq": subq, "index": i, "total": n, "found": len(findings)})
        return findings


# ─────────────────────────────────────────────────────────────────────────────
# The pipeline
# ─────────────────────────────────────────────────────────────────────────────
async def run_deep_research(agent, session, question: str, on_event=None):
    """Drive the four-role deep-research pipeline. Returns
    (answer_text, ['decompose','research','verify','synthesize'])."""

    # ── 1) DECOMPOSE (director) ──────────────────────────────────────────────
    await _emit(on_event, {"type": "stage", "stage": "decompose", "status": "start"})

    known_facts: list[str] = []
    try:
        nodes = await session.graph.search(question, limit=6)
        known_facts = [getattr(n, "content", "") for n in nodes if getattr(n, "content", "")]
        if known_facts:
            session.ledger.record_recall("\n".join(known_facts))  # reuse credited honestly
    except Exception:  # noqa: BLE001
        known_facts = []

    director_sys = (
        "You are a research director. Break the user's question into 3-6 "
        "independent, concretely searchable sub-questions that together fully "
        "cover it. Each must be self-contained and answerable by a web search. "
        "Return ONLY a JSON array of strings."
    )
    director_user = question
    if known_facts:
        block = "\n".join(f"- {f}" for f in known_facts)[:3000]
        director_user = (f"{question}\n\nAlready known (from prior research — do NOT "
                         f"re-ask these, focus on the gaps):\n{block}")
    subqs_raw = await _llm(agent, director_sys, director_user, effort=2)
    parsed = _loads_loose(subqs_raw)
    subqs = [str(s).strip() for s in parsed if isinstance(s, (str, int, float)) and str(s).strip()] \
        if isinstance(parsed, list) else []
    if not subqs:
        subqs = [question]                       # graceful fallback on a parse miss
    subqs = subqs[:_MAX_SUBQ]
    if len(subqs) < _MIN_SUBQ and subqs != [question]:
        # too few but parseable — keep them; never pad with junk
        pass

    await _emit(on_event, {"type": "stage", "stage": "decompose", "status": "done",
                           "subquestions": subqs})

    # ── 2) FAN-OUT sub-researchers (parallel, bounded) ───────────────────────
    sem = asyncio.Semaphore(_FANOUT)
    n = len(subqs)
    batches = await asyncio.gather(
        *[_research_subq(agent, session, sq, i, n, sem, on_event)
          for i, sq in enumerate(subqs)],
        return_exceptions=True,
    )
    findings: list[dict] = []
    for b in batches:
        if isinstance(b, Exception):
            logger.debug("sub-researcher failed: %s", b)
            continue
        findings.extend(b)

    # ── 3) ADVERSARIAL VERIFY ────────────────────────────────────────────────
    await _emit(on_event, {"type": "stage", "stage": "verify", "status": "start",
                           "count": len(findings)})
    verified: list[dict] = []
    dropped = 0
    if findings:
        # Recover each claim's source excerpt from the page cache (free reuse, NOT
        # credited — stays honest). Only claims whose source text we can actually
        # quote are adversarially tested; a claim we can't quote (cache-key miss) is
        # AUTO-KEPT rather than dropped — it already passed the supported-only
        # extraction, so a retrieval gap on our side must not refute it.
        def _excerpt(url: str) -> str:
            cache = session._page_cache
            for k in (url, url.rstrip("/"), url.split("#", 1)[0], url.split("#", 1)[0].rstrip("/")):
                t = cache.get(k)
                if t:
                    return t[:_VERIFY_EXCERPT]
            return ""

        lines = []
        idx_map: dict[int, int] = {}      # batch-line index -> findings index
        auto_keep: list[dict] = []
        for fi, f in enumerate(findings):
            excerpt = _excerpt(f["source_url"])
            if not excerpt:
                auto_keep.append(f)       # no quotable text -> don't risk a false refute
                continue
            li = len(lines)
            idx_map[li] = fi
            lines.append(
                f'[{li}] CLAIM: {f["claim"]}\n     SOURCE_URL: {f["source_url"]}\n'
                f"     SOURCE_TEXT: {excerpt}"
            )
        verdict_by_idx: dict[int, bool] = {}
        if lines:
            verify_sys = (
                "You are an adversarial fact-checker. For each numbered claim, try to "
                "REFUTE it using ONLY the quoted SOURCE_TEXT. A claim is supported ONLY "
                "if the source text explicitly states it; if the text is vague or does "
                "not state the claim, it is NOT supported. Return ONLY JSON: "
                '[{"index": <int>, "supported": <bool>, "reason": "..."}].'
            )
            verify_user = "\n\n".join(lines)[:16000] + "\n\nReturn the JSON array now."
            parsed = _loads_loose(await _llm(agent, verify_sys, verify_user, effort=2))
            if isinstance(parsed, list):
                for v in parsed:
                    if isinstance(v, dict) and "index" in v:
                        try:
                            verdict_by_idx[int(v["index"])] = bool(v.get("supported", True))
                        except Exception:  # noqa: BLE001
                            continue
        verified.extend(auto_keep)
        for li, fi in idx_map.items():
            # degrade safe: a claim the verifier didn't rule on stays kept.
            if verdict_by_idx.get(li, True) if verdict_by_idx else True:
                verified.append(findings[fi])
            else:
                dropped += 1
    await _emit(on_event, {"type": "stage", "stage": "verify", "status": "done",
                           "supported": len(verified), "dropped": dropped})

    # ── 4) SYNTHESIZE (streamed, cited) ──────────────────────────────────────
    await _emit(on_event, {"type": "stage", "stage": "synthesize", "status": "start",
                           "findings": len(verified)})

    # Number findings off the STABLE citation index so [n] matches the sources panel.
    block_lines: list[str] = []
    for f in verified:
        cite_n = session.cite("", f["source_url"]) if f.get("source_url") else 0
        if cite_n:
            block_lines.append(f"[{cite_n}] {f['claim']}")
        else:
            block_lines.append(f"- {f['claim']}")
    findings_block = "\n".join(block_lines) if block_lines else "(no verified findings)"

    synth_sys = (
        "Write a precise, well-structured answer to the user's question using ONLY "
        "the verified findings below. Use inline [n] citations matching the numbers "
        "provided — never renumber them. Note any contradictions or gaps explicitly. "
        "Do not invent facts, numbers, or sources. If the findings are insufficient, "
        "say so plainly."
    )
    synth_user = (f"Question: {question}\n\nVerified findings (cite as [n]):\n"
                  f"{findings_block}\n\nWrite the answer now.")

    answer = ""
    streamed = False
    try:
        async for chunk in agent.llm.chat_stream(
            [Message(role="system", content=synth_sys),
             Message(role="user", content=synth_user)],
            effort=2,
        ):
            piece = getattr(chunk, "content", "") or ""
            if piece:
                streamed = True
                answer += piece
                await _emit(on_event, {"type": "token", "text": piece})
    except Exception as exc:  # noqa: BLE001
        logger.debug("deep_mode synth stream failed: %s", exc)

    if not answer.strip():
        # Fallback: non-streamed single call, emitted as one token.
        answer = await _llm(agent, synth_sys, synth_user, effort=2)
        if answer:
            await _emit(on_event, {"type": "token", "text": answer})
        streamed = False

    await _emit(on_event, {"type": "stage", "stage": "synthesize", "status": "done"})
    logger.info("deep mode: %d subqs, %d findings, %d verified, streamed=%s",
                len(subqs), len(findings), len(verified), streamed)

    return answer.strip(), ["decompose", "research", "verify", "synthesize"]
