"""Reproduce a paper metric by running its command in a container.

Pass the same ReproState instance to all reproduction tools in an agent. Loading a
spec (load_paper_spec or generate_spec) must happen first; validation (validate_metric)
reads the value this tool stores in state.data["reproduced"].

Each value comes with `matched_line`, the output line it was read from, so a wrong
number (an early training step, another column) is visible. Reproduced values are also
written to reproduced_record.json in Team 2's metrics-record format.
"""
import json
import os
import time

from langchain_core.tools import tool

from . import docker_ops
from .extract import locate
from .spec_gen import PATTERN_TODO
from .state import ReproState

LOG_TAIL = 4000


def _tail(text: str, n: int = LOG_TAIL) -> str:
    # Apptainer prints a block of harmless mount warnings on every run; drop them so the
    # excerpt shows the program's own output.
    text = "\n".join(l for l in text.splitlines() if not l.startswith(("WARNING: Skipping mount", "WARNING: nv files",
                                                                      "WARNING: Overriding HOME")))
    return text if len(text) <= n else "...[truncated]...\n" + text[-n:]


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


def do_reproduce_metric(state: ReproState, metric: str, container: str = "") -> dict:
    spec = state.spec
    if metric not in spec.metrics:
        return {"status": "ERROR", "message": f"unknown metric '{metric}'", "available": list(spec.metrics)}
    m = spec.metrics[metric]
    container = container or spec.image_tag()
    command = m.command or spec.run_command
    if not command:
        return {"status": "ERROR", "message": "no run command in spec (run.command or metric.command)"}

    outputs = state.paper_dir("outputs")
    cached = state.cached_run(container, command)
    if cached:
        log_path, reused = cached["log"], True
        with open(log_path, errors="replace") as f:
            out = f.read()
    else:
        reused = False
        log_path = os.path.join(state.paper_dir("logs"), f"run-{state.run_key(container, command)}.log")
        volumes = {spec.resolve(h): c for h, c in spec.volumes.items()}
        t0 = time.time()
        try:
            rc, out = docker_ops.run(container, command, log_path=log_path, outputs_dir=outputs,
                                     gpu=spec.gpu, env=spec.env, volumes=volumes, workdir=spec.workdir,
                                     shm_size=spec.shm_size, timeout=spec.timeout_minutes * 60)
        except docker_ops.DockerError as e:
            return {"status": "ERROR", "message": str(e)}
        state.record_run(container, command, log_path, rc, time.time() - t0)
        if rc != 0:
            return {"status": "RUN_FAILED", "returncode": rc, "command": command, "log": log_path,
                    "log_tail": _tail(out)}

    if m.file:
        fpath = os.path.join(outputs, m.file)
        if not os.path.exists(fpath):
            return {"status": "ERROR", "message": f"metric file not produced: {fpath}", "log_tail": _tail(out)}
        with open(fpath, errors="replace") as f:
            source_text = f.read()
    else:
        source_text = out

    value, line = locate(source_text, m)
    if value is None:
        if m.pattern == PATTERN_TODO:
            message = ("the run finished, but this metric's pattern has not been written yet. Find the line "
                       "with the final result in the run log (read_file with search on the log path), then "
                       "set pattern (and occurrence) with update_spec_metric.")
        else:
            message = f"pattern {m.pattern!r} (after={m.after!r}) matched nothing"
        return {"status": "NOT_FOUND", "message": message, "command": command, "log": log_path,
                "log_tail": _tail(source_text)}
    state.data["reproduced"][state.key(metric)] = {"value": value, "container": container, "command": command,
                                                  "log": log_path, "matched_line": line,
                                                  "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    state.save()
    write_reproduced_record(state)
    return {"status": "OK", "metric": metric, "value": value, "matched_line": line,
            "reused_previous_run": reused, "log": log_path}


def write_reproduced_record(state: ReproState) -> None:
    """Reproduced values in Team 2's record format ({metric name: number}), next to the run logs,
    so their record tools can read them."""
    spec = state.spec
    record = {}
    for name, m in spec.metrics.items():
        rep = state.data["reproduced"].get(state.key(name))
        record[m.record_key or name] = rep["value"] if rep else None
    path = os.path.join(state.paper_dir(), "reproduced_record.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=4)
    os.replace(tmp, path)


def make_tools(state: ReproState) -> list:
    """Create reproduce_metric bound to shared session state."""
    @tool
    def reproduce_metric(metric: str, container: str = "") -> str:
        """Run the active paper's command inside `container` (a running container name, or an image
        tag - defaults to the spec's image) and parse `metric` from the output.
        Runs are cached: metrics produced by the same command reuse the previous run.
        May take hours for training runs. Returns the parsed value with `matched_line` (the output
        line it came from), or the failing log tail."""
        return _dump(do_reproduce_metric(state, metric, container))

    return [reproduce_metric]
