"""AitherSearch — multi-engine keyless web search for the research agent.

A research agent is only as good as what it can retrieve, so this fans every query
out across SEVERAL keyless engines CONCURRENTLY, dedupes by URL, and ranks by
ENGINE AGREEMENT (a URL surfaced by more independent engines is more trustworthy and
ranks higher) plus position. No API keys, no accounts — only the query leaves the
machine. Every engine degrades gracefully and the search never raises.

Engines (all keyless):
  - DuckDuckGo via the maintained `ddgs` library (structured, robust)
  - a rotating pool of public SearXNG instances (meta-search over Google/Bing/…)
  - Startpage (Google results, parseable HTML SERP)
  - DuckDuckGo HTML endpoint (last-resort scrape, no extra dep)

Return shape (back-compatible with the old single-engine extract, + `engines`):
    {"query", "results":[{"title","url","snippet","source"}], "provider", "count", "engines":[...]}

Dependency-light: httpx + lxml + ddgs, all already required by the pack.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

logger = logging.getLogger("deep_research.aithersearch")

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")
_HEADERS = {"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"}

# Public SearXNG instances (HTML scrape — the JSON API is blocked on most). Many are
# dead/empty at any moment, so we try a bounded slice concurrently and keep whatever
# answers. The first instance that yields hits floats to the front for later calls.
SEARX_POOL = [
    "https://opnxng.com", "https://search.mdosch.de", "https://etsi.me",
    "https://s.trung.fun", "https://priv.au", "https://searx.tiekoetter.com",
    "https://baresearch.org", "https://search.rhscz.eu", "https://paulgo.io",
]
_SEARX_OK: list[str] = []          # learned-good instances, front of the pool

# tiny cross-call TTL cache so a repeated query (follow-ups, multi-angle overlap)
# doesn't re-hit the network — pure savings, no staleness risk at this horizon.
_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 600.0


# --------------------------------------------------------------------------- lxml
def _html(text: str):
    from lxml import html as lh
    return lh.fromstring(text)


def _txt(el) -> str:
    return re.sub(r"\s+", " ", (el.text_content() if el is not None else "")).strip()


# ----------------------------------------------------------------------- engines
def _search_ddgs(query: str, limit: int) -> list[dict[str, Any]]:
    """DuckDuckGo via the ddgs/duckduckgo_search library (synchronous)."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            return []
    out: list[dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=limit):
                out.append({"title": (r.get("title") or "").strip(),
                            "url": r.get("href") or r.get("url") or "",
                            "snippet": (r.get("body") or "").strip()[:400],
                            "source": "duckduckgo"})
                if len(out) >= limit:
                    break
    except Exception as exc:  # noqa: BLE001
        logger.debug("ddgs failed: %s", exc)
    return out


async def _search_ddg_html(client: httpx.AsyncClient, query: str, limit: int) -> list[dict[str, Any]]:
    """DuckDuckGo HTML endpoint — keyless, no library needed."""
    out: list[dict[str, Any]] = []
    try:
        r = await client.get("https://html.duckduckgo.com/html/", params={"q": query})
        links = re.findall(r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r.text, re.S)
        snips = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.S)
        for i, (url, title) in enumerate(links[:limit]):
            if "uddg=" in url:
                try:
                    url = unquote(parse_qs(urlparse(url).query).get("uddg", [url])[0])
                except (ValueError, KeyError):
                    pass
            out.append({"title": re.sub(r"<[^>]+>", "", title).strip(),
                        "url": url,
                        "snippet": re.sub(r"<[^>]+>", "", snips[i]).strip()[:400] if i < len(snips) else "",
                        "source": "duckduckgo-html"})
    except Exception as exc:  # noqa: BLE001
        logger.debug("ddg-html failed: %s", exc)
    return out


async def _search_searxng(client: httpx.AsyncClient, query: str, limit: int) -> list[dict[str, Any]]:
    """A public SearXNG instance (meta-search). Tries a bounded slice of the pool."""
    pool = _SEARX_OK + [i for i in SEARX_POOL if i not in _SEARX_OK]
    for inst in pool[:4]:
        try:
            r = await client.get(inst.rstrip("/") + "/search",
                                  params={"q": query, "language": "en"})
            if r.status_code != 200 or "result" not in r.text:
                continue
            doc = _html(r.text)
            out: list[dict[str, Any]] = []
            for art in doc.xpath('//article[contains(@class,"result")] | //div[contains(@class,"result")]'):
                a = art.xpath('.//h3/a[@href] | .//a[contains(@class,"url_header")][@href] | .//a[@href]')
                if not a:
                    continue
                href = a[0].get("href", "")
                if not href.startswith("http"):
                    continue
                p = art.xpath('.//p[contains(@class,"content")] | .//p')
                out.append({"title": _txt(a[0]), "url": href,
                            "snippet": (_txt(p[0]) if p else "")[:400], "source": "searxng"})
                if len(out) >= limit:
                    break
            if out:
                if inst in _SEARX_OK:
                    _SEARX_OK.remove(inst)
                _SEARX_OK.insert(0, inst)
                return out
        except Exception as exc:  # noqa: BLE001
            logger.debug("searxng %s failed: %s", inst, exc)
            continue
    return []


async def _search_startpage(client: httpx.AsyncClient, query: str, limit: int) -> list[dict[str, Any]]:
    """Startpage — proxies Google and still serves a parseable HTML SERP."""
    out: list[dict[str, Any]] = []
    try:
        r = await client.get("https://www.startpage.com/sp/search",
                             params={"query": query, "language": "english"})
        if r.status_code != 200:
            return out
        doc = _html(r.text)
        for d in doc.xpath('//div[contains(@class,"result")] | //div[contains(@class,"w-gl__result")]'):
            a = d.xpath('.//a[contains(@class,"result-link")][@href] | .//a[contains(@class,"result-title")][@href]')
            if not a:
                continue
            href = a[0].get("href", "")
            if not href.startswith("http") or "startpage.com" in urlparse(href).netloc:
                continue
            p = d.xpath('.//p[contains(@class,"description")] | .//p')
            out.append({"title": _txt(a[0]), "url": href,
                        "snippet": (_txt(p[0]) if p else "")[:400], "source": "startpage"})
            if len(out) >= limit:
                break
    except Exception as exc:  # noqa: BLE001
        logger.debug("startpage failed: %s", exc)
    return out


# -------------------------------------------------------------- merge + rank
def _merge_rank(per_engine: list[list[dict]], limit: int) -> list[dict]:
    """Dedupe by URL and rank by ENGINE AGREEMENT (how many independent engines found
    it) then best position. A URL three engines agree on outranks a lone top hit."""
    agg: dict[str, dict] = {}
    for results in per_engine:
        for pos, r in enumerate(results):
            u = (r.get("url") or "").strip().rstrip("/")
            if not u:
                continue
            a = agg.get(u)
            if a is None:
                agg[u] = {**r, "url": u, "_engines": {r.get("source", "")}, "_best": pos}
            else:
                a["_engines"].add(r.get("source", ""))
                a["_best"] = min(a["_best"], pos)
                if len(r.get("snippet") or "") > len(a.get("snippet") or ""):
                    a["snippet"] = r["snippet"]   # keep the richest snippet
    ranked = sorted(agg.values(), key=lambda r: (len(r["_engines"]), -r["_best"]), reverse=True)
    out = []
    for r in ranked[:limit]:
        engines = sorted(e for e in r.pop("_engines") if e)
        r.pop("_best", None)
        r["source"] = ",".join(engines) or r.get("source", "")
        out.append(r)
    return out


# ----------------------------------------------------------------------- entry
async def search(query: str, limit: int = 6) -> dict[str, Any]:
    """Run a keyless multi-engine web search. Returns AitherSearch-shaped results,
    ranked by cross-engine agreement. Never raises."""
    query = (query or "").strip()
    if not query:
        return {"query": query, "results": [], "provider": "none", "count": 0, "engines": []}

    now = time.time()
    ck = f"{query}::{limit}"
    hit = _CACHE.get(ck)
    if hit and now - hit[0] < _CACHE_TTL:
        return {**hit[1], "cached": True}

    per = max(limit, 6)
    async with httpx.AsyncClient(timeout=12.0, follow_redirects=True, headers=_HEADERS) as client:
        results = await asyncio.gather(
            asyncio.to_thread(_search_ddgs, query, per),        # sync lib, off the loop
            _search_searxng(client, query, per),
            _search_startpage(client, query, per),
            return_exceptions=True,
        )
        engine_lists = [r for r in results if isinstance(r, list)]
        # if the structured engines all came up empty, fall back to the HTML scrape
        if not any(engine_lists):
            engine_lists = [await _search_ddg_html(client, query, per)]

    merged = _merge_rank(engine_lists, limit)
    engines = sorted({e for r in merged for e in (r.get("source") or "").split(",") if e})
    out = {"query": query, "results": merged, "provider": "multi" if len(engines) > 1 else (engines[0] if engines else "none"),
           "count": len(merged), "engines": engines}
    _CACHE[ck] = (now, out)
    if len(_CACHE) > 256:                                       # cheap bound
        for k in sorted(_CACHE, key=lambda k: _CACHE[k][0])[:64]:
            _CACHE.pop(k, None)
    return out
