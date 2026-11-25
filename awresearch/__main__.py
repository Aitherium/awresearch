"""`python -m awresearch` — a way in that does not need the console script.

A blocked console-script shim (Smart App Control judges each unsigned exe on its
own reputation) must leave another way in. The awdesk browser's Research mode
runs this CLI, and a host whose shim is blocked had no way to run a report.
"""

import sys

from awresearch.cli import main

if __name__ == "__main__":
    sys.exit(main())
