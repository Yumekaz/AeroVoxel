"""Strict ID validation for path-sensitive API parameters.

job_id and case_id are interpolated into filesystem paths. Reject anything that
is not a plain UUID / catalog-style token so ``..`` and separators cannot escape
asset directories.
"""

from __future__ import annotations

import re

from fastapi import HTTPException

# Upload jobs always use uuid.uuid4()
_JOB_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
# Demo / flow case tokens, e.g. sports_car_v1, sphere_3d_v1
_CASE_ID_RE = re.compile(r"^[a-zA-Z0-9_]{1,64}$")


def require_safe_job_id(job_id: str) -> str:
    if not job_id or not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id")
    return job_id


def require_safe_case_id(case_id: str) -> str:
    if not case_id or not _CASE_ID_RE.fullmatch(case_id):
        raise HTTPException(status_code=400, detail="Invalid case_id")
    return case_id
