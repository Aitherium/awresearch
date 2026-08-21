"""Source trust — cheap, offline, URL-first authority + freshness scoring.

A research answer is only as good as the sources behind it, so every registered
source gets a CHEAP trust stamp: a tiered domain-authority read (string-only, from
the registrable domain / TLD) plus a freshness read (a publish/update date parsed
from the URL path or the page text, decayed by age). No network, no LLM, no deps
beyond the stdlib (re / datetime / urllib.parse). Every function degrades to a
neutral value and NEVER raises — trust scoring must never break a research turn.

Public API:
  domain_authority(url)      -> {"domain","tier","score","reason"}
  parse_date(text, url="")   -> (iso_str|None, age_days|None)
  freshness(text="", url="") -> {"date","age_days","score"}
  trust_score(url, text="")  -> {"domain","tier","authority","freshness","date","age_days","score"}
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

# ── domain-authority tiers (curated, string-only) ─────────────────────────────
# High-authority TLD suffixes (institutional / governmental / academic).
_HIGH_TLDS = (".gov", ".mil", ".edu", ".int", ".gov.uk", ".ac.uk", ".edu.au",
              ".gov.au", ".gc.ca")

# Curated primary / institutional hosts (peer-reviewed press, standards bodies,
# multilateral institutions, primary databases). Score 0.95.
_HIGH_HOSTS = {
    "nature.com", "science.org", "sciencemag.org", "nejm.org", "thelancet.com",
    "cell.com", "pnas.org", "arxiv.org", "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov", "who.int", "sec.gov", "europa.eu", "ietf.org",
    "iso.org", "nist.gov", "imf.org", "worldbank.org",
}
# High-quality news wires + scholarly indexes. Score 0.85.
_HIGH_NEWS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "semanticscholar.org", "crossref.org",
}
# Reputable mainstream / trade press + pro engineering orgs + tertiary reference
# (Wikipedia is a strong starting point but a tertiary source, not primary). Score 0.6.
_MEDIUM_HOSTS = {
    "nytimes.com", "wsj.com", "ft.com", "bloomberg.com", "theguardian.com",
    "economist.com", "arstechnica.com", "wired.com", "theverge.com",
    "techcrunch.com", "ieee.org", "wikipedia.org",
}
# Open-platform / user-generated / social — low authority. Score 0.3.
_LOW_HOSTS = {
    "medium.com", "substack.com", "blogspot.com", "wordpress.com", "reddit.com",
    "quora.com", "tumblr.com", "facebook.com", "x.com", "twitter.com",
    "pinterest.com",
}

# second-to-last labels that signal a multi-label public suffix (co.uk, ac.uk…)
_PUBLIC_2LD = {"co", "ac", "gov", "org", "edu", "com", "net", "gc"}


def _registrable_domain(url: str) -> tuple[str, str]:
    """Return (registrable_domain, full_host) from a URL — string ops only.

    Handles the common multi-label public suffixes we care about (co.uk, ac.uk,
    gov.uk, gov.au, edu.au, gc.ca) by keeping three labels when the second-to-last
    is a known public-suffix label. Returns ("", host) on any failure.
    """
    try:
        host = url if "//" in url else "//" + url
        host = (urlsplit(host).netloc or "").lower().strip()
        if "@" in host:                       # strip userinfo
            host = host.rsplit("@", 1)[-1]
        host = host.split(":", 1)[0]          # strip port
        if host.startswith("www."):
            host = host[4:]
        if not host or "." not in host:
            return "", host
        labels = host.split(".")
        if len(labels) >= 3 and labels[-2] in _PUBLIC_2LD and len(labels[-1]) <= 3:
            dom = ".".join(labels[-3:])
        else:
            dom = ".".join(labels[-2:])
        return dom, host
    except Exception:  # noqa: BLE001
        return "", ""


def domain_authority(url: str) -> dict:
    """Tiered authority from the registrable domain / TLD only (cheap, url-only).

    Returns {"domain","tier","score","reason"}; tier ∈ high|medium|low|unknown.
    Pure string ops — never raises, never touches the network.
    """
    dom, host = _registrable_domain(str(url or ""))
    if not host:
        return {"domain": "", "tier": "unknown", "score": 0.45, "reason": "default"}
    # TLD / suffix wins first (an arbitrary .gov host is high regardless of name).
    for suf in _HIGH_TLDS:
        if host.endswith(suf):
            return {"domain": dom, "tier": "high", "score": 0.95, "reason": "tld"}
    if dom in _HIGH_HOSTS or host in _HIGH_HOSTS:
        return {"domain": dom, "tier": "high", "score": 0.95, "reason": "host"}
    if dom in _HIGH_NEWS or host in _HIGH_NEWS:
        return {"domain": dom, "tier": "high", "score": 0.85, "reason": "host"}
    if dom in _MEDIUM_HOSTS or host in _MEDIUM_HOSTS:
        return {"domain": dom, "tier": "medium", "score": 0.6, "reason": "host"}
    if dom in _LOW_HOSTS or host in _LOW_HOSTS:
        return {"domain": dom, "tier": "low", "score": 0.3, "reason": "host"}
    # A generic .org is NOT inherently trustworthy (anyone can register one), so it
    # reads as the neutral default — not a free medium-trust bump.
    return {"domain": dom, "tier": "unknown", "score": 0.45, "reason": "default"}


# ── date parsing ──────────────────────────────────────────────────────────────
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_RE_URL_DATE = re.compile(r"/(20\d{2})/(\d{1,2})(?:/(\d{1,2}))?/")
_RE_ISO = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_MONTH_ALT = "|".join(_MONTHS)
# 'Published/Updated/...' within ~40 chars of a 'Mon DD, YYYY' or 'DD Mon YYYY'.
_RE_LABEL_MDY = re.compile(
    r"(?:published|updated|posted|last\s+modified)[^0-9A-Za-z]{0,40}?"
    r"(" + _MONTH_ALT + r")[a-z]*\.?\s+(\d{1,2}),?\s+(20\d{2})", re.IGNORECASE)
_RE_LABEL_DMY = re.compile(
    r"(?:published|updated|posted|last\s+modified)[^0-9A-Za-z]{0,40}?"
    r"(\d{1,2})\s+(" + _MONTH_ALT + r")[a-z]*\.?\s+(20\d{2})", re.IGNORECASE)


def _mk(year: int, month: int, day: int) -> datetime | None:
    try:
        now = datetime.now(timezone.utc)
        if not (2000 <= year <= now.year):
            return None
        month = min(max(month, 1), 12)
        day = min(max(day, 1), 28) if day else 1
        return datetime(year, month, day, tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def parse_date(text: str, url: str = "") -> tuple[str | None, int | None]:
    """Best-effort publish/update date -> (iso_str|None, age_days|None).

    Order of preference: (1) a date in the URL path (/YYYY/MM[/DD]/), (2) an ISO
    date in the page text, (3) a 'Published/Updated …' labelled date. The first
    plausible date with a year in [2000, now] wins. Never raises.
    """
    dt: datetime | None = None
    try:
        u = str(url or "")
        t = str(text or "")
        m = _RE_URL_DATE.search(u)
        if m:
            dt = _mk(int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))
        if dt is None:
            m = _RE_ISO.search(t)
            if m:
                dt = _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if dt is None:
            m = _RE_LABEL_MDY.search(t)
            if m:
                dt = _mk(int(m.group(3)), _MONTHS.get(m.group(1).lower(), 0), int(m.group(2)))
        if dt is None:
            m = _RE_LABEL_DMY.search(t)
            if m:
                dt = _mk(int(m.group(3)), _MONTHS.get(m.group(2).lower(), 0), int(m.group(1)))
    except Exception:  # noqa: BLE001
        dt = None
    if dt is None:
        return None, None
    age = (datetime.now(timezone.utc) - dt).days
    return dt.date().isoformat(), max(0, age)


def freshness(text: str = "", url: str = "") -> dict:
    """Parse a date and decay it to a 0..1 freshness score.

    Step decay: <=30d 1.0, <=180d 0.85, <=365d 0.7, <=3y 0.5, else 0.3. An
    unknown date is NEUTRAL 0.5 (we don't punish undated primary sources).
    """
    iso, age = parse_date(text, url)
    if age is None:
        return {"date": None, "age_days": None, "score": 0.5}
    if age <= 30:
        score = 1.0
    elif age <= 180:
        score = 0.85
    elif age <= 365:
        score = 0.7
    elif age <= 3 * 365:
        score = 0.5
    else:
        score = 0.3
    return {"date": iso, "age_days": age, "score": score}


def trust_score(url: str, text: str = "") -> dict:
    """Combined trust = 0.7*authority + 0.3*freshness — the single call tools.py
    uses to stamp a source. Returns the merged authority+freshness dict."""
    a = domain_authority(url)
    f = freshness(text, url)
    score = round(0.7 * a["score"] + 0.3 * f["score"], 3)
    return {
        "domain": a["domain"], "tier": a["tier"], "authority": a["score"],
        "freshness": f["score"], "date": f["date"], "age_days": f["age_days"],
        "score": score,
    }
