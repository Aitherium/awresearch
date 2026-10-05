"""Command-line interface for awresearch.

Contract (stable; a GUI may rely on it):

    awresearch --question Q [--depth standard|deep] [--output markdown|json]
               [--out-file F] [--max-sources N] [--model M] [--events] [-v]

* stdout: the report (Markdown or JSON) -- empty when ``--out-file`` is given.
* stderr: with ``--events``, one JSON object per line per progress event,
  ``{"phase": "<phase>", "message": "<text>"}``; on failure a single line
  ``awresearch: <reason>``. Other stderr lines are diagnostics (not JSON).
* exit: 0 report written; 1 no LLM reachable or the run failed; 2 usage error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__

logger = logging.getLogger("awresearch.cli")


def _fail(reason: str) -> int:
    print(f"awresearch: {reason}", file=sys.stderr, flush=True)
    return 1


def _event_writer(enabled: bool):
    if not enabled:
        return None

    def write(ev: dict) -> None:
        line = json.dumps({"phase": str(ev.get("phase", "")),
                           "message": str(ev.get("message", ""))})
        sys.stderr.write(line + "\n")
        sys.stderr.flush()

    return write


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entry point. Returns the process exit code."""
    # GENERATED doctor intercept (gen_aw_doctor.py) -- do not edit
    _dv = locals().get("argv")
    if (_dv if _dv is not None else __import__("sys").argv[1:])[:1] == ["doctor"]:
        from ._doctor import report
        return report()
    # GENERATED repo-state intercept (gen_aw_doctor.py) -- do not edit
    try:
        from awgit import state as _aw_state
    except Exception:
        _aw_state = None
    if _aw_state is not None:
        _sv = locals().get("argv")
        if _aw_state.cli_banner(_sv if _sv is not None else __import__("sys").argv[1:]):
            return 0
    parser = argparse.ArgumentParser(
        prog="awresearch",
        description="Ask a research question, get a cited report you can check.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--question", help="The research question to answer")
    parser.add_argument(
        "--depth",
        choices=["standard", "deep"],
        default="standard",
        help="standard = one search/read/extract pass; deep = sub-questions, "
             "parallel research, adversarial verification (default: standard)",
    )
    parser.add_argument(
        "--output",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument("--out-file", type=Path, help="Write the report to this file")
    parser.add_argument("--max-sources", type=int, default=10,
                        help="Page-read budget for the run (default: 10)")
    parser.add_argument("--model", help="Model to request (default: the backend's own)")
    parser.add_argument("--events", action="store_true",
                        help='Write progress as JSON lines on stderr: {"phase", "message"}')
    parser.add_argument("-v", "--verbose", action="store_true", help="Diagnostic logging")

    args = parser.parse_args(argv)
    if not (args.question or "").strip():
        parser.error("--question is required")
    if args.max_sources < 1:
        parser.error("--max-sources must be >= 1")

    for stream in (sys.stdout, sys.stderr):
        try:  # a report quotes pages in any script; a cp1252 console must not crash it
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError) as exc:  # not a text stream (embedded, redirected)
            logger.debug("cannot switch %r to utf-8: %s", stream, exc)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        import adk  # noqa: F401
    except ImportError:
        return _fail("the agent engine is not installed: pip install awdk")

    from .api import LLMUnavailableError, Researcher

    researcher = Researcher(model=args.model)
    try:
        report = asyncio.run(researcher.research(
            args.question, depth=args.depth, max_sources=args.max_sources,
            on_event=_event_writer(args.events),
        ))
    except LLMUnavailableError as exc:
        return _fail(str(exc))
    except KeyboardInterrupt:
        return _fail("interrupted")
    except Exception as exc:  # noqa: BLE001 -- one line for the user, detail under -v
        logger.debug("research failed", exc_info=True)
        first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        return _fail(f"research failed: {type(exc).__name__}: {first}")

    if args.output == "json":
        out = json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n"
    else:
        out = report.markdown()

    if args.out_file:
        try:
            args.out_file.parent.mkdir(parents=True, exist_ok=True)
            args.out_file.write_text(out, encoding="utf-8")
        except OSError as exc:
            return _fail(f"cannot write {args.out_file}: {exc}")
    else:
        sys.stdout.write(out)
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
