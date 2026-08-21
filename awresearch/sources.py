"""Specialized & recovery sources — keyless adapters for primary research.

Web search gets you the open web; PRIMARY research needs the scholarly record and
the ability to recover sources that vanished. This module adds two keyless (no API
key, no account) capabilities, each mirroring aithersearch.py's contract — module
_UA/_HEADERS, every source degrades to []/{} on failure, NOTHING ever raises:

  scholarly_search(query, limit) — arXiv (Atom/XML) + Semantic Scholar (JSON) +
      Crossref (JSON), fetched CONCURRENTLY and merged/deduped by DOI then title,
      enriching empty fields across providers. Returns papers with DOIs.
  wayback_lookup(url, limit) — Internet Archive Wayback CDX + availability API:
      recover the closest live snapshot of a dead/moved URL plus a capture timeline
      (snapshot URLs are directly fetchable by the existing fetch_url/fetch_many).

Dependency-light: httpx + lxml + json/stdlib only.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger("deep_research.sources")

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")
_HEADERS = {"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"}

_WS = re.compile(r"\s+")


def _collapse(s: str) -> str:
    return _WS.sub(" ", (s or "")).strip()


# ─────────────────────────────────────────────────────────────── scholarly
async def _arxiv(client: httpx.AsyncClient, query: str, limit: int) -> list[dict]:
    """arXiv Atom API — parsed with lxml from BYTES (str trips the XML decl)."""
    try:
        from lxml import etree
        r = await client.get("http://export.arxiv.org/api/query",
                             params={"search_query": f"all:{query}",
                                     "start": 0, "max_results": limit})
        if r.status_code != 200:
            return []
        root = etree.fromstring(r.content)            # bytes, not .text
        ns = {"atom": "http://www.w3.org/2005/Atom",
              "arxiv": "http://arxiv.org/schemas/atom"}
        out: list[dict] = []
        for e in root.findall(".//atom:entry", ns):
            title = _collapse("".join(e.findtext("atom:title", "", ns)))
            if not title:
                continue
            abstract = _collapse("".join(e.findtext("atom:summary", "", ns)))
            published = e.findtext("atom:published", "", ns) or ""
            year = published[:4] if published[:4].isdigit() else ""
            authors = [_collapse(n.text or "")
                       for n in e.findall("atom:author/atom:name", ns)]
            url = _collapse(e.findtext("atom:id", "", ns))
            doi = _collapse(e.findtext("arxiv:doi", "", ns))
            out.append({"title": title, "abstract": abstract, "year": year,
                        "venue": "arXiv", "authors": [a for a in authors if a],
                        "url": url, "doi": doi, "source": "arxiv"})
            if len(out) >= limit:
                break
        return out
    except Exception as exc:  # noqa: BLE001
        logger.debug("arxiv failed: %s", exc)
        return []


async def _semantic_scholar(client: httpx.AsyncClient, query: str, limit: int) -> list[dict]:
    """Semantic Scholar Graph API (keyless, shared pool — may 429 -> [])."""
    try:
        r = await client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": min(limit, 100),
                    "fields": "title,abstract,year,venue,authors,externalIds,url"})
        if r.status_code != 200:
            return []
        data = r.json()
        out: list[dict] = []
        for p in data.get("data", []) or []:
            title = _collapse(p.get("title") or "")
            if not title:
                continue
            doi = _collapse((p.get("externalIds") or {}).get("DOI", "") or "")
            url = (p.get("url") or (f"https://doi.org/{doi}" if doi else
                   f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}"))
            out.append({"title": title,
                        "abstract": _collapse(p.get("abstract") or ""),
                        "year": str(p.get("year") or ""),
                        "venue": _collapse(p.get("venue") or ""),
                        "authors": [a.get("name", "") for a in (p.get("authors") or [])],
                        "url": url, "doi": doi, "source": "semantic_scholar"})
        return out
    except Exception as exc:  # noqa: BLE001
        logger.debug("semantic_scholar failed: %s", exc)
        return []


def _strip_jats(s: str) -> str:
    """Crossref abstracts are JATS XML (<jats:p>…) — strip tags, collapse ws."""
    if not s:
        return ""
    try:
        return _collapse(re.sub(r"<[^>]+>", " ", s))
    except Exception:  # noqa: BLE001
        return ""


async def _crossref(client: httpx.AsyncClient, query: str, limit: int) -> list[dict]:
    """Crossref works API (polite pool via mailto, no key)."""
    try:
        r = await client.get(
            "https://api.crossref.org/works",
            params={"query": query, "rows": limit,
                    "select": "DOI,title,author,issued,container-title,abstract",
                    "mailto": "research@aitherium.com"})
        if r.status_code != 200:
            return []
        items = (r.json().get("message", {}) or {}).get("items", []) or []
        out: list[dict] = []
        for it in items:
            titles = it.get("title") or []
            title = _collapse(titles[0]) if titles else ""
            if not title:
                continue
            doi = _collapse(it.get("DOI", "") or "")
            year = ""
            try:
                year = str(it["issued"]["date-parts"][0][0])
            except Exception:  # noqa: BLE001
                year = ""
            containers = it.get("container-title") or []
            venue = _collapse(containers[0]) if containers else ""
            authors = [_collapse(f"{a.get('given', '')} {a.get('family', '')}")
                       for a in (it.get("author") or [])]
            out.append({"title": title,
                        "abstract": _strip_jats(it.get("abstract", "") or ""),
                        "year": year, "venue": venue,
                        "authors": [a for a in authors if a],
                        "url": f"https://doi.org/{doi}" if doi else "",
                        "doi": doi, "source": "crossref"})
            if len(out) >= limit:
                break
        return out
    except Exception as exc:  # noqa: BLE001
        logger.debug("crossref failed: %s", exc)
        return []


async def _pubmed(client: httpx.AsyncClient, query: str, limit: int) -> list[dict]:
    """PubMed via NCBI E-utilities (keyless): esearch -> PMIDs, esummary -> details.
    Adds biomedical coverage to the scholarly merge. esummary carries no abstract, so
    other providers fill it; the PMID URL + DOI still anchor the citation."""
    try:
        r = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmode": "json",
                    "retmax": min(limit, 20), "sort": "relevance"})
        if r.status_code != 200:
            return []
        ids = ((r.json().get("esearchresult") or {}).get("idlist") or [])
        if not ids:
            return []
        r2 = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
        if r2.status_code != 200:
            return []
        res = r2.json().get("result", {}) or {}
        out: list[dict] = []
        for pid in res.get("uids", []) or []:
            p = res.get(pid) or {}
            title = _collapse(p.get("title") or "")
            if not title:
                continue
            doi = ""
            for aid in p.get("articleids", []) or []:
                if aid.get("idtype") == "doi":
                    doi = _collapse(aid.get("value", "") or "")
                    break
            year = (p.get("pubdate") or "")[:4]
            authors = [_collapse(a.get("name", "")) for a in (p.get("authors") or [])]
            out.append({"title": title, "abstract": "",
                        "year": year if year.isdigit() else "",
                        "venue": _collapse(p.get("fulljournalname") or p.get("source") or ""),
                        "authors": [a for a in authors if a],
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                        "doi": doi, "source": "pubmed"})
        return out
    except Exception as exc:  # noqa: BLE001
        logger.debug("pubmed failed: %s", exc)
        return []


def _norm_doi(doi: str) -> str:
    d = (doi or "").lower().strip()
    d = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", d)
    return d


def _norm_title(title: str) -> str:
    return _collapse(re.sub(r"[^a-z0-9 ]+", " ", (title or "").lower()))


def _merge_papers(lists: list[list[dict]], limit: int) -> list[dict]:
    """Dedupe by normalized DOI then normalized title; enrich empty fields across
    providers (a Crossref DOI can fill an arXiv hit), keep the longest abstract,
    rank by provider agreement then presence of abstract/year."""
    agg: dict[str, dict] = {}
    for results in lists:
        for r in results:
            doi = _norm_doi(r.get("doi", ""))
            key = f"doi:{doi}" if doi else f"ti:{_norm_title(r.get('title', ''))}"
            if not key or key in ("ti:", "doi:"):
                continue
            cur = agg.get(key)
            if cur is None:
                agg[key] = {**r, "_providers": {r.get("source", "")}}
                continue
            cur["_providers"].add(r.get("source", ""))
            for f in ("year", "venue", "doi", "url"):
                if not cur.get(f) and r.get(f):
                    cur[f] = r[f]
            if not cur.get("authors") and r.get("authors"):
                cur["authors"] = r["authors"]
            if len(r.get("abstract") or "") > len(cur.get("abstract") or ""):
                cur["abstract"] = r["abstract"]
    ranked = sorted(
        agg.values(),
        key=lambda r: (len(r["_providers"]), bool(r.get("abstract")), bool(r.get("year"))),
        reverse=True)
    out: list[dict] = []
    for r in ranked[:limit]:
        provs = sorted(p for p in r.pop("_providers") if p)
        r["source"] = ",".join(provs)
        out.append(r)
    return out


async def scholarly_search(query: str, limit: int = 8) -> dict[str, Any]:
    """Keyless scholarly literature search across arXiv + Semantic Scholar +
    Crossref, merged/deduped by DOI then title. Never raises."""
    query = (query or "").strip()
    if not query:
        return {"query": query, "papers": [], "count": 0, "providers": []}
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True,
                                     headers=_HEADERS) as client:
            results = await asyncio.gather(
                _arxiv(client, query, limit),
                _semantic_scholar(client, query, limit),
                _crossref(client, query, limit),
                _pubmed(client, query, limit),
                return_exceptions=True,
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("scholarly_search client failed: %s", exc)
        return {"query": query, "papers": [], "count": 0, "providers": []}
    lists = [r for r in results if isinstance(r, list)]
    providers = sorted({r[0]["source"] for r in lists if r})
    papers = _merge_papers(lists, limit)
    return {"query": query, "papers": papers, "count": len(papers),
            "providers": providers}


# ─────────────────────────────────────────────────────────────── recovery
def _ts_to_dt(ts: str) -> str:
    """Wayback 14-digit timestamp -> 'YYYY-MM-DD HH:MM:SS'."""
    ts = (ts or "").strip()
    if len(ts) >= 14 and ts[:14].isdigit():
        return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]} {ts[8:10]}:{ts[10:12]}:{ts[12:14]}"
    return ts


async def wayback_lookup(url: str, limit: int = 10) -> dict[str, Any]:
    """Recover archived snapshots of a dead/moved URL via the Internet Archive
    Wayback Machine (CDX timeline + closest-available). Never raises."""
    url = str(url or "").strip()
    if "." not in url:                       # not URL-like; standalone, no coupling
        return {"url": url, "captures": [], "count": 0,
                "note": "pass a concrete URL to recover its archived snapshots"}
    captures: list[dict] = []
    closest: dict = {"available": False, "snapshot_url": "", "timestamp": ""}
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True,
                                     headers=_HEADERS) as client:
            cdx, avail = await asyncio.gather(
                client.get("http://web.archive.org/cdx/search/cdx",
                           params={"url": url, "output": "json", "limit": limit,
                                   "filter": "statuscode:200",
                                   "collapse": "timestamp:8"}),
                client.get("http://archive.org/wayback/available",
                           params={"url": url}),
                return_exceptions=True,
            )
        if isinstance(cdx, httpx.Response) and cdx.status_code == 200:
            try:
                rows = cdx.json()
            except Exception:  # noqa: BLE001
                rows = []
            for row in rows[1:] if rows else []:   # row[0] is the header
                try:
                    ts, original = row[1], row[2]
                    captures.append({
                        "timestamp": ts, "datetime": _ts_to_dt(ts),
                        "snapshot_url": f"http://web.archive.org/web/{ts}/{original}",
                        "statuscode": row[4], "mimetype": row[3]})
                except Exception:  # noqa: BLE001
                    continue
        if isinstance(avail, httpx.Response) and avail.status_code == 200:
            try:
                snap = ((avail.json().get("archived_snapshots") or {})
                        .get("closest") or {})
                if snap:
                    closest = {"available": bool(snap.get("available")),
                               "snapshot_url": snap.get("url", ""),
                               "timestamp": snap.get("timestamp", "")}
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("wayback_lookup failed: %s", exc)
    stamps = [c["timestamp"] for c in captures if c.get("timestamp")]
    return {"url": url, "closest": closest, "captures": captures,
            "count": len(captures),
            "first": min(stamps) if stamps else "",
            "last": max(stamps) if stamps else ""}


# ─────────────────────────────────────────────── filings (SEC EDGAR, keyless)
# SEC requires a descriptive User-Agent with a contact address or it 403s.
_SEC_HEADERS = {"User-Agent": "Aitherium Deep Research research@aitherium.com",
                "Accept-Encoding": "gzip, deflate"}


async def filings_search(query: str, limit: int = 10) -> dict[str, Any]:
    """SEC EDGAR full-text search of public-company filings (10-K/10-Q/8-K/S-1/…),
    keyless. Returns filings with company, form type, date, and a direct document
    URL (fetchable by fetch_url). Use for primary-source corporate/financial facts.
    Never raises."""
    query = str(query or "").strip()
    if not query:
        return {"query": query, "filings": [], "count": 0}
    out: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True,
                                     headers=_SEC_HEADERS) as client:
            r = await client.get("https://efts.sec.gov/LATEST/search-index",
                                 params={"q": query})
            if r.status_code != 200:
                return {"query": query, "filings": [], "count": 0}
            hits = ((r.json().get("hits") or {}).get("hits") or [])
        for h in hits[:limit]:
            s = h.get("_source", {}) or {}
            adsh = s.get("adsh", "") or ""
            ciks = s.get("ciks") or []
            cik = (ciks[0] if ciks else "").lstrip("0")
            doc = str(h.get("_id", "")).partition(":")[2]
            url = ""
            if adsh and doc and cik:
                url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                       f"{adsh.replace('-', '')}/{doc}")
            roots = s.get("root_forms") or []
            form = s.get("file_type") or (roots[0] if roots else "")
            out.append({"company": "; ".join(s.get("display_names") or []),
                        "form": form, "date": s.get("file_date", ""),
                        "accession": adsh, "url": url, "source": "sec_edgar"})
    except Exception as exc:  # noqa: BLE001
        logger.debug("sec edgar failed: %s", exc)
    return {"query": query, "filings": out, "count": len(out)}


# ─────────────────────────────────────────────── code (GitHub repos, keyless)
async def code_search(query: str, limit: int = 10) -> dict[str, Any]:
    """GitHub repository search (keyless; unauthenticated rate limit ~10/min),
    sorted by stars. Returns repos with name, description, stars, language, URL and
    last-push date. Use for implementations, tools, and OSS state of the art.
    Never raises."""
    query = str(query or "").strip()
    if not query:
        return {"query": query, "repos": [], "count": 0}
    out: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True,
                                     headers={**_HEADERS, "Accept": "application/vnd.github+json"}) as client:
            r = await client.get("https://api.github.com/search/repositories",
                                 params={"q": query, "sort": "stars", "order": "desc",
                                         "per_page": min(limit, 20)})
            if r.status_code != 200:
                return {"query": query, "repos": [], "count": 0,
                        "note": "github rate-limited or no results" if r.status_code == 403 else ""}
            for it in (r.json().get("items") or [])[:limit]:
                out.append({"name": it.get("full_name", ""),
                            "description": _collapse(it.get("description") or ""),
                            "stars": it.get("stargazers_count", 0),
                            "language": it.get("language") or "",
                            "url": it.get("html_url", ""),
                            "updated": (it.get("pushed_at") or "")[:10],
                            "source": "github"})
    except Exception as exc:  # noqa: BLE001
        logger.debug("github failed: %s", exc)
    return {"query": query, "repos": out, "count": len(out)}


# ─────────────────────────────────────────── entities (Wikidata, keyless)
# Wikimedia's robot policy 403s generic browser UAs; it wants a descriptive
# tool UA with a contact URL/email.
_WIKI_HEADERS = {"User-Agent": "AitheriumDeepResearch/1.0 (https://aitherium.com; research@aitherium.com)",
                 "Accept": "application/json"}


async def wikidata_lookup(query: str, limit: int = 8) -> dict[str, Any]:
    """Wikidata entity search (keyless) — resolve a name/term to canonical entities
    with stable IDs (Q-numbers), labels and descriptions. Use to disambiguate and
    ground entities (people, companies, places, concepts) before researching them.
    Never raises."""
    query = str(query or "").strip()
    if not query:
        return {"query": query, "entities": [], "count": 0}
    out: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True,
                                     headers=_WIKI_HEADERS) as client:
            r = await client.get("https://www.wikidata.org/w/api.php",
                                 params={"action": "wbsearchentities", "search": query,
                                         "language": "en", "format": "json",
                                         "limit": min(limit, 20)})
            if r.status_code != 200:
                return {"query": query, "entities": [], "count": 0}
            for e in (r.json().get("search") or [])[:limit]:
                qid = e.get("id", "")
                out.append({"id": qid, "label": _collapse(e.get("label") or ""),
                            "description": _collapse(e.get("description") or ""),
                            "url": e.get("concepturi") or
                            (f"https://www.wikidata.org/wiki/{qid}" if qid else ""),
                            "source": "wikidata"})
    except Exception as exc:  # noqa: BLE001
        logger.debug("wikidata failed: %s", exc)
    return {"query": query, "entities": out, "count": len(out)}
