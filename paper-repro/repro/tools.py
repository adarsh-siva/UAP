"""Reproduction tools.

Core tools (the reproduction loop):
    docker_build(dockerfile)            build an image for the paper's code
    reproduce_metric(metric, container) run the paper's command in the container, parse the metric
    validate_metric(metric)             compare the reproduced value with the recorded value

Supporting tools (so an agent can onboard a new paper):
    load_paper_spec, inspect_repo, read_file, write_file, read_paper
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from typing import Optional

from langchain_core.tools import tool

from . import docker_ops
from .extract import compare, extract_value
from .state import ReproState

LOG_TAIL = 4000


def _tail(text: str, n: int = LOG_TAIL) -> str:
    return text if len(text) <= n else "...[truncated]...\n" + text[-n:]


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


# --------------------------------------------------------------------------- core logic

def do_load_spec(state: ReproState, spec_path: str) -> dict:
    spec = state.set_spec(spec_path)
    return {
        "paper": spec.name, "repo": spec.repo, "image": spec.image_tag(),
        "dockerfile": spec.resolve(spec.dockerfile), "run_command": spec.run_command,
        "gpu": spec.gpu, "volumes": spec.volumes, "notes": spec.notes,
        "metrics": {n: {"recorded": m.recorded, "tolerance": f"{m.tolerance} ({m.tolerance_type})",
                        "source": m.source, "command": m.command or "(paper run command)"}
                    for n, m in spec.metrics.items()},
    }


def do_docker_build(state: ReproState, dockerfile: str, tag: str = "", context: str = "") -> dict:
    spec = state.spec if state.data.get("spec_path") else None
    df = spec.resolve(dockerfile) if spec else os.path.abspath(os.path.expanduser(dockerfile))
    if not os.path.isfile(df):
        return {"status": "ERROR", "message": f"Dockerfile not found: {df}"}
    if context:
        ctx = spec.resolve(context) if spec else os.path.abspath(context)
    elif spec and df == spec.resolve(spec.dockerfile):
        ctx = spec.resolve(spec.context)
    else:
        ctx = os.path.dirname(df)
    tag = tag or (spec.image_tag() if spec else f"repro-{os.path.basename(os.path.dirname(df)).lower()}:latest")
    log = os.path.join(state.paper_dir("logs") if spec else state.workspace, f"build-{tag.replace(':', '_').replace('/', '_')}.log")
    t0 = time.time()
    try:
        rc, out = docker_ops.build(df, ctx, tag, log)
    except docker_ops.DockerError as e:
        return {"status": "ERROR", "message": str(e)}
    state.data["images"][tag] = {"dockerfile": df, "context": ctx, "ok": rc == 0, "log": log,
                                 "seconds": round(time.time() - t0, 1)}
    state.save()
    if rc != 0:
        return {"status": "FAILED", "image": tag, "log": log, "log_tail": _tail(out)}
    return {"status": "OK", "image": tag, "seconds": round(time.time() - t0, 1), "log": log}


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

    value = extract_value(source_text, m)
    if value is None:
        return {"status": "NOT_FOUND", "message": f"pattern {m.pattern!r} (after={m.after!r}) matched nothing",
                "command": command, "log": log_path, "log_tail": _tail(source_text)}
    state.data["reproduced"][state.key(metric)] = {"value": value, "container": container, "command": command,
                                                  "log": log_path, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    state.save()
    return {"status": "OK", "metric": metric, "value": value, "reused_previous_run": reused, "log": log_path}


EDITABLE_METRIC_FIELDS = {"recorded", "source", "tolerance", "tolerance_type", "pattern", "after",
                          "occurrence", "scale", "file", "command"}


def do_update_metric(state: ReproState, metric: str, updates: dict) -> dict:
    """Change named fields of one metric in the active spec file, leaving everything else as it was."""
    from ruamel.yaml import YAML

    from .spec import load_spec

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
    output_fields = set(updates) & {"pattern", "after", "occurrence", "scale", "file", "command"}
    rejected = None
    if output_fields:
        # These fields describe how to read the program's output, not the paper. Only accept a
        # change that demonstrably extracts a number from a real run of this metric's command.
        # A rejected output change never blocks the paper fields sent alongside it.
        from dataclasses import replace
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
                if extract_value(f.read(), trial) is None:
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
    return {"status": "OK", "metric": metric, "updated": updates}


def do_validate_metric(state: ReproState, metric: str) -> dict:
    spec = state.spec
    if metric not in spec.metrics:
        return {"status": "ERROR", "message": f"unknown metric '{metric}'", "available": list(spec.metrics)}
    rep = state.data["reproduced"].get(state.key(metric))
    if rep is None:
        return {"status": "ERROR", "message": f"'{metric}' has not been reproduced yet - call reproduce_metric first"}
    result = {"metric": metric, **compare(rep["value"], spec.metrics[metric])}
    state.data["validations"][state.key(metric)] = result
    state.save()
    return result


# --------------------------------------------------------------------------- LangChain tools

def make_tools(state: ReproState, writable_roots: list) -> list:
    roots = [os.path.abspath(r) for r in writable_roots]

    @tool
    def load_paper_spec(spec_path: str) -> str:
        """Load a paper's reproduction spec (YAML) and make it the active paper.
        Returns the paper's image tag, Dockerfile, run command and metrics with their recorded values.
        Call this before docker_build / reproduce_metric / validate_metric."""
        try:
            return _dump(do_load_spec(state, spec_path))
        except Exception as e:  # spec errors are for the agent to fix
            return _dump({"status": "ERROR", "message": f"{type(e).__name__}: {e}"})

    @tool
    def docker_build(dockerfile: str, tag: str = "", context: str = "") -> str:
        """Build a Docker image from a Dockerfile. Paths are relative to the active spec's folder.
        `tag` defaults to the spec's image tag; `context` defaults to the spec's build context.
        Returns status OK with the image tag, or FAILED with the tail of the build log."""
        return _dump(do_docker_build(state, dockerfile, tag, context))

    @tool
    def reproduce_metric(metric: str, container: str = "") -> str:
        """Run the active paper's command inside `container` (a running container name, or an image
        tag - defaults to the spec's image) and parse `metric` from the output.
        Runs are cached: metrics produced by the same command reuse the previous run.
        May take hours for training runs. Returns the parsed value or the failing log tail."""
        return _dump(do_reproduce_metric(state, metric, container))

    @tool
    def validate_metric(metric: str) -> str:
        """Compare the reproduced value of `metric` with its recorded value from the spec, within
        the spec's tolerance. Returns PASS / FAIL / NO_RECORD with the difference."""
        return _dump(do_validate_metric(state, metric))

    @tool
    def update_metric(metric: str, recorded: Optional[float] = None, source: str = "",
                      tolerance: Optional[float] = None, pattern: str = "", after: str = "") -> str:
        """Edit one metric in the active spec, changing only the fields you pass and keeping the
        rest of the file exactly as it is. Use this - not write_file.
        - `recorded`, `source`, `tolerance`: what the PAPER reports for this metric.
        - `pattern`, `after`: regexes for reading OUR RUN's output (what the code prints), not the
          paper. Leave them out unless reproduce_metric returned NOT_FOUND; such edits are only
          accepted if they find a number in the latest run log."""
        updates = {k: v for k, v in {"recorded": recorded, "tolerance": tolerance}.items() if v is not None}
        updates.update({k: v for k, v in {"source": source, "pattern": pattern, "after": after}.items() if v})
        if not updates:
            return _dump({"status": "ERROR", "message": "pass at least one field to change"})
        try:
            return _dump(do_update_metric(state, metric, updates))
        except RuntimeError as e:
            return _dump({"status": "ERROR", "message": str(e)})

    @tool
    def inspect_repo(repo_url: str) -> str:
        """Clone a git repository (shallow) into the workspace and summarize it: file tree,
        README, and dependency files. Use it to write a Dockerfile and spec for a new paper."""
        name = re.sub(r"\.git$", "", repo_url.rstrip("/").split("/")[-1])
        dest = os.path.join(state.workspace, "repos", name)
        if not os.path.isdir(dest):
            p = subprocess.run(["git", "clone", "-q", "--depth", "1", repo_url, dest],
                               capture_output=True, text=True, timeout=600)
            if p.returncode != 0:
                return _dump({"status": "ERROR", "message": p.stderr[-2000:]})
        tree = []
        for root, dirs, files in os.walk(dest):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            depth = os.path.relpath(root, dest).count(os.sep)
            if depth > 2:
                continue
            for f in files:
                tree.append(os.path.relpath(os.path.join(root, f), dest))
        summary = {"path": dest, "files": sorted(tree)[:300]}
        for fname in ["README.md", "readme.md", "README.rst", "requirements.txt", "environment.yml",
                      "setup.py", "pyproject.toml"]:
            fp = os.path.join(dest, fname)
            if os.path.exists(fp):
                with open(fp, errors="replace") as f:
                    summary[fname] = f.read()[:6000]
        return _dump(summary)

    @tool
    def read_file(path: str, start_line: int = 1, max_lines: int = 300) -> str:
        """Read lines from a text file (e.g. a cloned repo's training script, to find its
        command-line flags and how it prints metrics)."""
        try:
            with open(os.path.expanduser(path), errors="replace") as f:
                lines = f.readlines()
        except OSError as e:
            return f"ERROR: {e}"
        chunk = lines[start_line - 1:start_line - 1 + max_lines]
        return "".join(f"{i}\t{l}" for i, l in enumerate(chunk, start_line)) + \
            (f"\n... ({len(lines)} lines total)" if len(lines) > start_line - 1 + max_lines else "")

    @tool
    def write_file(path: str, content: str) -> str:
        """Write a text file (Dockerfile, spec YAML, helper script). Only allowed under the
        workspace or papers directories."""
        full = os.path.abspath(os.path.expanduser(path))
        if not any(full == r or full.startswith(r + os.sep) for r in roots):
            return f"ERROR: writes are only allowed under {roots}"
        os.makedirs(os.path.dirname(full), exist_ok=True)
        previous = None
        if os.path.exists(full):
            with open(full) as f:
                previous = f.read()
        with open(full, "w") as f:
            f.write(content)
        if os.path.basename(full) == "spec.yaml":
            from .spec import load_spec
            try:
                load_spec(full)
            except Exception as e:
                if previous is None:
                    os.remove(full)
                else:
                    with open(full, "w") as f:
                        f.write(previous)
                return (f"ERROR: not saved - the spec does not load ({type(e).__name__}: {e}). "
                        "To change an existing spec, use update_metric instead of rewriting it.")
        return f"wrote {len(content)} chars to {full}"

    @tool
    def read_paper(source: str, search: str = "", start_line: int = 0, num_lines: int = 60) -> str:
        """Read a paper PDF (URL or local path) as numbered lines of plain text.
        - No `search` and no `start_line`: returns an outline - every table/figure caption and
          section heading with its line number. Start here.
        - `start_line` (no search): returns `num_lines` lines from there (e.g. a table found in
          the outline).
        - `search`: a case-insensitive regex; returns matching lines with context (only lines at
          or after `start_line`, if given).
        Tables are extracted as whitespace-separated rows with NO '|' separators, e.g.
        'Origin 93.86 100.00 94.60 93.89'. Search for a caption ('Table 1') or a row label,
        not for table markup. Column headers are usually one or two lines above the rows."""
        from pypdf import PdfReader
        cache_dir = os.path.join(state.workspace, "paper_text")
        os.makedirs(cache_dir, exist_ok=True)
        key = re.sub(r"[^A-Za-z0-9]+", "_", source)[-80:]
        txt_path = os.path.join(cache_dir, key + ".txt")
        if not os.path.exists(txt_path):
            pdf_path = source
            if re.match(r"https?://", source):
                pdf_path = os.path.join(cache_dir, key + ".pdf")
                p = subprocess.run(["curl", "-sSL", "-o", pdf_path, source], capture_output=True, text=True, timeout=300)
                if p.returncode != 0:
                    return f"ERROR downloading: {p.stderr[-1000:]}"
            try:
                text = "\n".join((pg.extract_text() or "") for pg in PdfReader(pdf_path).pages)
            except Exception as e:
                return f"ERROR reading PDF: {e}"
            with open(txt_path, "w") as f:
                f.write(text)
        with open(txt_path) as f:
            lines = f.read().splitlines()
        caption_rx = re.compile(r"^\s*(Table|Figure|Fig\.)\s*\d+\s*[:.]|^\s*\d+(\.\d+)*\s+[A-Z][A-Za-z -]{2,60}$")
        outline = [f"{i}: {l.strip()[:150]}" for i, l in enumerate(lines) if caption_rx.search(l)]

        if start_line and not search:
            lo = max(0, start_line)
            # A span that starts just inside a table backs up to its caption, so the column
            # headers always come with the rows.
            for j in range(lo - 1, max(-1, lo - 9), -1):
                if re.match(r"^\s*(Table|Figure|Fig\.)\s*\d+", lines[j]):
                    lo = j
                    break
            span = lines[lo:lo + max(1, min(num_lines, 200))]
            return "\n".join(f"{i}: {l}" for i, l in enumerate(span, lo)) + f"\n({len(lines)} lines total)"
        if not search:
            return (f"{len(lines)} lines. Outline (captions and headings with line numbers); "
                    f"read a span with start_line=<n>:\n" + "\n".join(outline))[:15000]
        try:
            rx = re.compile(search, re.IGNORECASE)
        except re.error as e:
            return f"ERROR: invalid regex: {e}"
        hits = [i for i, l in enumerate(lines) if i >= max(0, start_line) and rx.search(l)]
        if not hits:
            tables = [o for o in outline if re.search(r":\s*(Table|Figure|Fig\.)", o)]
            return (f"no lines match {search!r}. Remember tables have no '|' separators. "
                    f"Captions in this paper (read one with start_line=<n>):\n" + "\n".join(tables))[:15000]
        # Context windows around hits, merged when they touch or nearly touch, so a table row
        # between two windows is never silently dropped.
        blocks = []
        for i in hits[:40]:
            lo, hi = max(0, i - 3), min(len(lines), i + 12)
            if blocks and lo <= blocks[-1][1] + 3:
                blocks[-1][1] = max(blocks[-1][1], hi)
            else:
                blocks.append([lo, hi])
        out = []
        for lo, hi in blocks:
            out.extend(f"{j}: {lines[j]}" for j in range(lo, hi))
            out.append("---")
        return "\n".join(out)[:15000] or "no matches"

    return [load_paper_spec, docker_build, reproduce_metric, validate_metric, update_metric,
            inspect_repo, read_file, write_file, read_paper]
