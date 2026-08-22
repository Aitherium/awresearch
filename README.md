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

<!-- aither-ecosystem:start GENERATED from the ecosystem registry. Edits here are overwritten; change the registry instead. -->

## The aw family

Standalone tools that share one idea: **replace something you would otherwise have to _trust_ with something you can _check_.**

Each installs on its own, works offline, and needs no account.

| | instead of trusting | you check |
|---|---|---|
| [awdk](https://github.com/Aitherium/awdk) | a framework's idea of how your agents should run | one loop you can read, pointed at a backend you already pay for |
| [awskills](https://github.com/Aitherium/awskills) | that an agent knows your procedure | the procedure written down, versioned, and loadable by any agent |
| [awm](https://github.com/Aitherium/awm) | that memory stayed in its lane | tenant:user:project scopes, so a write cannot cross a boundary |
| [awnode](https://github.com/Aitherium/awnode) | a vendor's cloud with every prompt | a local gateway routing to backends you chose |
| [awgraph](https://github.com/Aitherium/awgraph) | that grep found everything | an AST + tree-sitter call graph an agent can traverse |
| [awgit](https://github.com/Aitherium/awgit) | that no one else is editing this file | a lease, refused at commit time if you do not hold it |
| [awseal](https://github.com/Aitherium/awseal) | that the artifact came from who you think | an Ed25519 seal — the key that verifies is not the key that forges |
| [awshare](https://github.com/Aitherium/awshare) | that the download is intact | content-addressed bundles, verified on fetch |
| [awnest](https://github.com/Aitherium/awnest) | that there is a person on the other end | a verdict with evidence, where "we could not tell" is not "yes" |
| [awnboard](https://github.com/Aitherium/awnboard) | a share link anyone who sees it can use | an invitation addressed to one person, for one gate, revocable |
| [awnix](https://github.com/Aitherium/awnix) | that the box is what you left it as | an immutable image you built, with atomic rollback |
| [awrecover](https://github.com/Aitherium/awrecover) | that the restore worked | a restore that fully lands or does not land at all |
| [awrelay](https://github.com/Aitherium/awrelay) | a SaaS in the middle of your agents | findings, alerts and coordination over your own transport |
| [awmail](https://github.com/Aitherium/awmail) | a mailbox somebody else can read | mail your agents send and receive over your own server |
| [awfind](https://github.com/Aitherium/awfind) | one vendor's idea of the web | results from whichever providers you configured |
| [awbrowse](https://github.com/Aitherium/awbrowse) | that the page said what you were told | the render, the DOM and the requests it made |
| [aitherkvcache](https://github.com/Aitherium/aitherkvcache) | a vendor's quantisation defaults | sub-byte KV cache kernels you can benchmark yourself |
| [AitherZero](https://github.com/Aitherium/AitherZero) | a pile of scripts nobody has numbered | numbered, discoverable automation with declarative playbooks |
| [AitherConnect](https://github.com/Aitherium/AitherConnect) | what a page tells your browser to do | a federated search and desktop bridge you host |
| [awreason](https://github.com/Aitherium/awreason) | a confident paragraph | the phases it went through, and every tool call it made to get there |
| [awrecurse](https://github.com/Aitherium/awrecurse) | that everything you pasted in was actually read | which slices it opened, and what it concluded from each |
| [awprism](https://github.com/Aitherium/awprism) | the first explanation that fits | the ranked alternatives, and the observation that separates them |
| [awrepl](https://github.com/Aitherium/awrepl) | what the agent believes the value is | the value, printed from the live session |
| **awresearch** _(you are here)_ | a summary of pages nobody opened | every claim against the source it came from |
| [awkno](https://github.com/Aitherium/awkno) | that the docs site is up, or that you remember the family | the whole ecosystem in your terminal, with no network at all |

[**awnix**](https://github.com/Aitherium/awnix) is the ground floor — A Linux you can hand to an agent — immutable base, capabilities included.

## The Aitherium ecosystem

Every repository here is public. Each publishes an `aither-manifest.json` beside its page, so any surface can read every sibling's — the network is browsable from any node in it.

| repo | what it is | pages |
|---|---|---|
| [awdk](https://github.com/Aitherium/awdk) | Build AI agent fleets — 3 lines, any backend, local or cloud | [docs](https://aitherium.github.io/awdk/) |
| [awskills](https://github.com/Aitherium/awskills) | Portable agent skills — self-contained procedures an agent loads on demand | [docs](https://aitherium.github.io/awskills/) |
| [awm](https://github.com/Aitherium/awm) | A portable, scoped agent memory | [docs](https://aitherium.github.io/awm/) |
| [awnode](https://github.com/Aitherium/awnode) | A lightweight local gateway — bridges your apps to the AI backends you chose | [docs](https://aitherium.github.io/awnode/) |
| [awrun](https://github.com/Aitherium/awrun) | A priority-aware queue and dispatcher for agentic runs and ad-hoc CI builds | [docs](https://aitherium.github.io/awrun/) |
| [awgraph](https://github.com/Aitherium/awgraph) | A semantic code graph for agents — AST + tree-sitter, call graphs | [docs](https://aitherium.github.io/awgraph/) |
| [awgit](https://github.com/Aitherium/awgit) | Semantic version control on top of git — edit-ops and leases | [docs](https://aitherium.github.io/awgit/) |
| [awseal](https://github.com/Aitherium/awseal) | Sign an artifact so a stranger can verify it | [docs](https://aitherium.github.io/awseal/) |
| [awshare](https://github.com/Aitherium/awshare) | Publish an artifact and fetch it back verified | [docs](https://aitherium.github.io/awshare/) |
| [awnest](https://github.com/Aitherium/awnest) | Prove there is a human before you let them into the nest | [docs](https://aitherium.github.io/awnest/) |
| [awnboard](https://github.com/Aitherium/awnboard) | A front gate you can put in front of anything, and hand someone the key to | [docs](https://aitherium.github.io/awnboard/) |
| [awnix](https://github.com/Aitherium/awnix) | A Linux you can hand to an agent — immutable base, capabilities included | [docs](https://aitherium.github.io/awnix/) |
| [awrecover](https://github.com/Aitherium/awrecover) | Labelled snapshots with an all-or-nothing restore | [docs](https://aitherium.github.io/awrecover/) |
| [awrelay](https://github.com/Aitherium/awrelay) | Portable agent messaging — findings, alerts, coordination | [docs](https://aitherium.github.io/awrelay/) |
| [awmail](https://github.com/Aitherium/awmail) | Give an agent an email address — send, and actually receive | [docs](https://aitherium.github.io/awmail/) |
| [awfind](https://github.com/Aitherium/awfind) | A portable search client — query, results, ranking | [docs](https://aitherium.github.io/awfind/) |
| [awbrowse](https://github.com/Aitherium/awbrowse) | A portable browser client — navigate, console, network, DOM, screenshot | [docs](https://aitherium.github.io/awbrowse/) |
| [awknowledge](https://github.com/Aitherium/awknowledge) | How to run a coding agent so the result survives — the laws, with evidence | [docs](https://aitherium.github.io/awknowledge/) |
| [aitherkvcache](https://github.com/Aitherium/aitherkvcache) | Near-optimal KV cache quantization for LLM inference — sub-byte compression | [docs](https://aitherium.github.io/aitherkvcache/) |
| [AitherZero](https://github.com/Aitherium/AitherZero) | PowerShell 7+ automation framework — numbered, self-describing scripts | [docs](https://aitherium.github.io/AitherZero/) |
| [AitherConnect](https://github.com/Aitherium/AitherConnect) | Browser extension — federated AI search, page context, and the Living OS overlay | [docs](https://aitherium.github.io/AitherConnect/) |
| [awreason](https://github.com/Aitherium/awreason) | A portable reasoning client — sessions, phases, thoughts, and the chain that produced the answer | [docs](https://aitherium.github.io/awreason/) |
| [awrecurse](https://github.com/Aitherium/awrecurse) | Answer a question over a context far larger than the window — recursively, with the trace kept | [docs](https://aitherium.github.io/awrecurse/) |
| [awprism](https://github.com/Aitherium/awprism) | Turn a failure into ranked hypotheses — and say what would confirm each one | [docs](https://aitherium.github.io/awprism/) |
| [awrepl](https://github.com/Aitherium/awrepl) | A REPL an agent can actually use — state that survives between turns | [docs](https://aitherium.github.io/awrepl/) |
| **awresearch** _(you are here)_ | Ask a research question, get a cited report you can check | [docs](https://aitherium.github.io/awresearch/) |
| [awkno](https://github.com/Aitherium/awkno) | The man page for the Aither World — every brick, stack and law, offline | [docs](https://aitherium.github.io/awkno/) |

<div id="aither-constellation" data-self="awresearch"></div>
<script src="aither-constellation.js"></script>

<!-- aither-ecosystem:end -->
