"""Print non-destructive diagnostics for the optional neural reconstruction runtime."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.recon.diagnostics import collect_recon_diagnostics


if __name__ == "__main__":
    print(json.dumps(collect_recon_diagnostics(), indent=2))
