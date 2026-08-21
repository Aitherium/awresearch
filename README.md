# awresearch

**Ask a research question, get a cited report you can check.**

A research agent that actually reads its sources — it searches, retrieves the real pages, extracts specific claims with citations, and refuses to state anything it didn't retrieve. Every claim in the report is traced back to where it came from.

This solves the plausible-document problem: an LLM asked to research something returns fluent, confident text whose claims were never verified against real sources. awresearch trades speed for honesty — it costs real API calls and real retrieval time, and you get a report where checking the facts is possible.

## Install

```bash
pip install awresearch
```

Requires Python 3.10+, `awdk` (the agent engine), and an LLM provider (Anthropic / OpenAI / DeepSeek / local Ollama).

## Quick start

```python
from awdk.agent import AitherAgent
from awdk.identity import Identity
from awdk.memory import Memory
from awresearch.api import Researcher, Report

# Set up your LLM key in the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)

# Create a Researcher with an awdk agent
identity = Identity(name="researcher", description="Research analyst")
memory = Memory(db_path=".data/researcher.db", agent_name="researcher")
agent = AitherAgent(name="researcher", identity=identity, memory=memory)

researcher = Researcher(llm_backend=agent.llm)

# Ask a question
report = await researcher.research(
    "What are the latest breakthroughs in solid-state batteries?",
    depth="standard",
)

# Check the results
print(f"Claims: {len(report.claims)}")
print(f"Sources: {len(report.sources)}")
print(f"All claims sourced? {all(c.is_sourced for c in report.claims)}")

# Export as Markdown with citations
markdown = report.markdown()
print(markdown)

# Or as structured data
import json
print(json.dumps(report.to_dict(), indent=2))
```

## API

### `Researcher(llm_backend, search_backend=None, artifacts_dir=None)`

Configurable research agent.

**Args:**
- `llm_backend` — an awdk LLMRouter or compatible (required)
- `search_backend` — callable returning search results. Defaults to AitherSearch (multi-engine web search)
- `artifacts_dir` — where to store temporary files. Defaults to `AITHER_DATA_DIR` env var

### `async Researcher.research(question, depth="standard", max_sources=10) -> Report`

Research a question and return a cited report.

**Args:**
- `question` — the research question
- `depth` — `"standard"` (one pass) or `"deep"` (iterative multi-angle)
- `max_sources` — maximum sources to fetch (default: 10)

### `Report`

Result of a research session: claims, sources, validation.

**Attributes:**
- `question` — the original question
- `claims` — list of `Claim` objects
- `sources` — list of `Source` objects (each referenced by claims)
- `research_depth` — "standard" or "deep"

**Methods:**
- `validate()` → list of validation issues (empty = valid)
- `to_dict()` → JSON-serializable dict
- `markdown()` → Markdown report with citations

### `Claim`

A single claim in the report.

**Attributes:**
- `text` — the claim
- `sources` — list of 1-based indices into `Report.sources`
- `is_sourced` — True if has at least one source
- `unsourced_reason` — if unsourced, why (e.g., "too general to cite")

### `Source`

A retrieved source: URL, title, freshness/trust metadata.

**Attributes:**
- `url` — the source URL
- `title` — page title
- `retrieved_at` — ISO timestamp
- `domain` — domain name
- `authority` — domain authority score (0-1)
- `freshness` — freshness score (0-1, based on publication date)
- `trust` — combined trust score

## Configuration

Set via environment variables:

- `AWRESEARCH_SEARCH_BACKEND` — search engine ("ddgs", "searxng", etc.). Default: multi-engine (AitherSearch)
- `AITHER_DATA_DIR` — where to store artifacts. Default: temp directory

## Limitations

- **Real API calls.** Every search, every page fetch costs an HTTP request. Research is not free.
- **Sources can be wrong.** This tool cross-checks claims against retrieved pages, but cannot verify that pages themselves are accurate. It is not a fact-checker.
- **Paywalls and access.** Some sources are behind paywalls or require authentication. The tool retrieves what is publicly accessible.
- **Speed.** Real research takes time. A deep-research turn may take 30–60 seconds per question.

## CLI

```bash
# Not yet implemented. See awresearch Python API above.
awresearch --question "Research question here" --output markdown
```

## Architecture

awresearch is built on:

- **awdk** — the ReAct agent loop, memory system, and LLM routing
- **AitherSearch** — multi-engine web search (DuckDuckGo, SearXNG, Startpage)
- **Scholarly sources** — arXiv, Semantic Scholar, Crossref (for academic papers)
- **Trust scoring** — domain authority and content freshness detection
- **Knowledge graph** — awdk's GraphMemory for conversation memory and fact recall

## License

Apache-2.0. See [LICENSE](LICENSE).

## Contributing

Contributions welcome. See [GitHub](https://github.com/Aitherium/awresearch).

---

**Built by Aitherium.** [GitHub](https://github.com/Aitherium) · [Docs](https://github.com/Aitherium/awresearch#readme)
