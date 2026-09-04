"""Non-destructive host diagnostics for neural reconstruction setup."""

from __future__ import annotations

import csv
import os
import platform
import subprocess
from typing import Any


def _nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        return {
            "available": False,
            "returncode": completed.returncode,
            "error": (completed.stderr or completed.stdout).strip()[:500],
        }
    rows = list(csv.reader(line for line in completed.stdout.splitlines() if line.strip()))
    gpus = [
        {"name": row[0].strip(), "memory_total_mb": row[1].strip(), "driver_version": row[2].strip()}
        for row in rows
        if len(row) >= 3
    ]
    return {"available": bool(gpus), "gpus": gpus}


def _windows_display_diagnostics() -> dict[str, Any] | None:
    if platform.system() != "Windows":
        return None
    # WMI is queried through PowerShell without a shell command assembled from
    # user input. Failure is reported, never treated as GPU availability.
    script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion,Status,ConfigManagerErrorCode | "
        "ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        return {"error": (completed.stderr or completed.stdout).strip()[:500]}
    try:
        import json

        value = json.loads(completed.stdout)
        return {"devices": value if isinstance(value, list) else [value]}
    except (ValueError, TypeError) as exc:
        return {"error": f"Could not parse display diagnostics: {exc}"}


def collect_recon_diagnostics() -> dict[str, Any]:
    """Collect runtime, NVIDIA CLI, and Windows display status without mutation."""
    from app.recon.capability import get_recon_capability

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "working_directory": os.getcwd(),
        "runtime": get_recon_capability(),
        "nvidia_smi": _nvidia_smi(),
        "windows_display_devices": _windows_display_diagnostics(),
    }
