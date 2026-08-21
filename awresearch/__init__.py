"""awresearch — ask a research question, get a cited report you can check.

A research question answered with real sources: the agent searches, reads the
actual pages, cross-checks claims, and returns a report where every claim carries
where it came from. No plausible fabrication — every fact is traced to a source.

Public API:
  - Researcher(...) — configured with search and completion backends
  - .research(question, ...) -> Report
  - Report.claims, .sources, .to_dict(), .markdown()
"""

from __future__ import annotations

from .api import Claim, Report, Researcher, Source

__version__ = "0.1.0"

__all__ = [
    "Researcher",
    "Report",
    "Claim",
    "Source",
]
