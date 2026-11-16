"""Pull a metric value out of run output and compare it with a recorded value."""
from __future__ import annotations

import re
from typing import Optional, Tuple

from .spec import MetricSpec

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def locate(text: str, m: MetricSpec) -> Tuple[Optional[float], str]:
    """Find the metric's number in `text`. Returns (value, the line it came from), or (None, "")."""
    text = ANSI.sub("", text)
    if m.after:
        anchors = list(re.finditer(m.after, text, re.MULTILINE))
        if not anchors:
            return None, ""
        text = text[anchors[-1].end():]
    matches = list(re.finditer(m.pattern, text, re.MULTILINE))
    if not matches:
        return None, ""
    if m.occurrence == "first":
        hit = matches[0]
    elif m.occurrence == "last":
        hit = matches[-1]
    else:
        idx = int(m.occurrence)
        if not -len(matches) <= idx < len(matches):
            return None, ""
        hit = matches[idx]
    group = 1 if hit.re.groups else 0
    try:
        value = float(hit.group(group)) * m.scale
    except (TypeError, ValueError):
        return None, ""
    start = text.rfind("\n", 0, hit.start(group)) + 1
    end = text.find("\n", hit.end(group))
    line = text[start:end if end != -1 else len(text)].strip()
    return value, line[:300]


def extract_value(text: str, m: MetricSpec) -> Optional[float]:
    return locate(text, m)[0]


def compare(reproduced: float, m: MetricSpec) -> dict:
    if m.recorded is None:
        return {"status": "NO_RECORD", "reproduced": reproduced,
                "message": f"metric '{m.name}' has no recorded value in the spec"}
    diff = reproduced - m.recorded
    if m.tolerance_type == "rel":
        allowed = abs(m.recorded) * m.tolerance
    else:
        allowed = m.tolerance
    ok = abs(diff) <= allowed + 1e-12
    return {
        "status": "PASS" if ok else "FAIL",
        "reproduced": reproduced,
        "recorded": m.recorded,
        "difference": round(diff, 6),
        "allowed_difference": round(allowed, 6),
        "tolerance": f"{m.tolerance} ({m.tolerance_type})",
        "source": m.source,
    }
