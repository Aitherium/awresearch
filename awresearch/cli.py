"""Command-line interface for awresearch."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("awresearch.cli")


def main():
    """Main CLI entry point."""
    # GENERATED doctor intercept (gen_aw_doctor.py) -- do not edit
    _dv = locals().get("argv")
    if (_dv if _dv is not None else __import__("sys").argv[1:])[:1] == ["doctor"]:
        from ._doctor import report
        return report()
    parser = argparse.ArgumentParser(
        prog="awresearch",
        description="Ask a research question, get a cited report you can check.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--question",
        help="The research question to answer",
    )
    parser.add_argument(
        "--depth",
        choices=["standard", "deep"],
        default="standard",
        help="Research depth (default: standard)",
    )
    parser.add_argument(
        "--output",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        help="Write output to file instead of stdout",
    )

    args = parser.parse_args()

    if not args.question:
        parser.print_help()
        print(
            "\nERROR: --question is required",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "awresearch CLI requires awdk integration; see README for full setup",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
