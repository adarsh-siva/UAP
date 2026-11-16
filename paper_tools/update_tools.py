"""update_spec_metric: edit one metric of the active paper's spec (Team 4's Update_metric).

This is the version the agent uses: it works on the YAML spec loaded into the shared
ReproState, keeps the rest of the file exactly as it was (comments, order, quoting), and
tests changes to how the run's output is read against the real run log before saving them.
The standalone JSON implementation is update_metric.py at the repository root.

Named update_spec_metric so it does not clash with Team 2's update_metric, which edits
their metrics record.
"""
import json
import os
import threading
import time
from dataclasses import replace
from typing import Optional

from langchain_core.tools import tool

from .extract import locate
from .reproduce_tools import write_reproduced_record
from .spec import load_spec
from .state import ReproState

EDITABLE_METRIC_FIELDS = {"recorded", "source", "tolerance", "tolerance_type", "pattern", "after",
                          "occurrence", "scale", "file", "command"}
OUTPUT_FIELDS = {"pattern", "after", "occurrence", "scale", "file", "command"}

# An agent may call several tools at once (parallel threads). Spec edits read the whole file,
# change one metric and write it back, so two at once would overwrite each other: they take
# turns. generate_spec (spec_gen_tools.py) uses the same lock.
SPEC_LOCK = threading.RLock()


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


def do_update_metric(state: ReproState, metric: str, updates: dict) -> dict:
    """Change named fields of one metric in the active spec file, leaving everything else as it was."""
    with SPEC_LOCK:
        return _update_metric_locked(state, metric, updates)


def _update_metric_locked(state: ReproState, metric: str, updates: dict) -> dict:
    from ruamel.yaml import YAML

    bad = set(updates) - EDITABLE_METRIC_FIELDS
    if bad:
        return {"status": "ERROR", "message": f"cannot edit {sorted(bad)}; editable: {sorted(EDITABLE_METRIC_FIELDS)}"}
    path = state.spec.path
    with open(path) as f:
        original = f.read()
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.representer.add_representer(
        type(None), lambda r, _: r.represent_scalar("tag:yaml.org,2002:null", "null"))
    doc = yaml.load(original)
    metrics = doc.get("metrics") or {}
    if metric not in metrics:
        return {"status": "ERROR", "message": f"unknown metric '{metric}'", "available": list(metrics)}
    output_fields = set(updates) & OUTPUT_FIELDS
    rejected = None
    new_value, new_log, new_line = None, None, ""
    if output_fields:
        # These fields describe how to read the program's output, not the paper. Only accept a
        # change that demonstrably extracts a number from a real run of this metric's command.
        # A rejected output change never blocks the paper fields sent alongside it.
        current = state.spec.metrics[metric]
        trial = replace(current, **{k: updates[k] for k in output_fields})
        command = trial.command or state.spec.run_command
        logs = [r["log"] for r in state.data["runs"].values()
                if r["command"] == command and r.get("returncode") == 0 and os.path.exists(r["log"])]
        if not logs:
            rejected = ("they describe how to read the run's output (not the paper), and there is no "
                        "successful run of this metric's command to test them against yet")
        else:
            with open(logs[-1], errors="replace") as f:
                new_value, new_line = locate(f.read(), trial)
            new_log = logs[-1]
            if new_value is None:
                rejected = f"nothing is found with them in the latest run output ({logs[-1]})"
        if rejected:
            updates = {k: v for k, v in updates.items() if k not in output_fields}
            if not updates:
                return {"status": "ERROR", "message": f"not changed {sorted(output_fields)}: {rejected}."}
    for k, v in updates.items():
        metrics[metric][k] = v
    with open(path, "w") as f:
        yaml.dump(doc, f)
    try:
        load_spec(path)
    except Exception as e:
        with open(path, "w") as f:
            f.write(original)
        return {"status": "ERROR", "message": f"edit rejected, spec left unchanged: {type(e).__name__}: {e}"}
    state.set_spec(path)
    if rejected:
        return {"status": "PARTIAL", "metric": metric, "saved": updates,
                "not_changed": sorted(output_fields),
                "message": f"saved {sorted(updates)}; kept the existing {sorted(output_fields)} because {rejected}. "
                           "The existing ones are fine to leave as they are."}
    if output_fields and new_value is not None:
        # The reading settings changed, so the old reproduced value and verdict are stale.
        key = state.key(metric)
        state.data["reproduced"][key] = {"value": new_value, "log": new_log, "matched_line": new_line,
                                         "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                         "note": "re-extracted after update_spec_metric"}
        state.data["validations"].pop(key, None)
        state.save()
        write_reproduced_record(state)
        return {"status": "OK", "metric": metric, "updated": updates,
                "value_from_latest_run": new_value, "matched_line": new_line,
                "message": "Saved. matched_line is the output line the number was read from: check it is the "
                           "final result you meant (not an early step, another column or another metric), "
                           "then call validate_metric."}
    return {"status": "OK", "metric": metric, "updated": updates}


def make_tools(state: ReproState) -> list:
    """Create update_spec_metric bound to shared session state."""
    @tool
    def update_spec_metric(metric: str, recorded: Optional[float] = None, source: str = "",
                           tolerance: Optional[float] = None, pattern: str = "", after: str = "",
                           occurrence: str = "") -> str:
        """Edit one metric in the active spec, changing only the fields you pass and keeping the
        rest of the file exactly as it is. Use this - not write_file. (Named update_spec_metric so it
        does not clash with Team 2's update_metric, which edits their metrics record.)
        - `recorded`, `source`, `tolerance`: what the PAPER reports for this metric.
        - `pattern`, `after`, `occurrence`: how to read OUR RUN's output (what the code prints),
          not the paper. `occurrence` picks which match to use when the pattern matches several
          times: "first", "last", or an index such as "-1". Change these when reproduce_metric
          returned NOT_FOUND or an implausible value; such edits are only accepted if they find a
          number in the latest run log, and the result reports the value and the line they read."""
        updates = {k: v for k, v in {"recorded": recorded, "tolerance": tolerance}.items() if v is not None}
        updates.update({k: v for k, v in {"source": source, "pattern": pattern, "after": after}.items() if v})
        if occurrence:
            occ = occurrence.strip().lower()
            if occ not in ("first", "last"):
                try:
                    occ = int(occ)
                except ValueError:
                    return _dump({"status": "ERROR", "message": "occurrence must be first, last, or an integer index"})
            updates["occurrence"] = occ
        if not updates:
            return _dump({"status": "ERROR", "message": "pass at least one field to change"})
        return _dump(do_update_metric(state, metric, updates))

    return [update_spec_metric]
