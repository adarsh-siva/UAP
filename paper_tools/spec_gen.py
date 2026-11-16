"""Work out how to run a paper's code, and write its spec.yaml.

The other teams supply the inputs: Team 1 the Dockerfile, Team 2 the metrics record (the
paper's values), Team 3 a scanner for command-line options. What only the repo's own code can
answer is worked out here, statically, without running anything:

  - which commands the repo documents, and which options its scripts take
  - where the code runs inside the container (from the Dockerfile)
  - where the code reads its data from, including paths hard-coded in the source
  - how it prints its results, and what it needs to run (wandb, GPU)

How to read a metric out of the output can only be settled from a real run, so the spec is
written with a placeholder pattern for each metric, to be set with update_spec_metric after
the first run.
"""
from __future__ import annotations

import ast
import json
import os
import posixpath
import re
import shlex
import sys
from pathlib import Path

import yaml

from .spec import load_spec, read_record

PATTERN_TODO = r"(?!x)x"          # matches nothing; reproduce_metric reports it as not written yet
SKIP_DIRS = {".git", "venv", "env", "node_modules", "__pycache__", "site-packages", "build", "dist"}
DATA_WORDS = re.compile(r"(^|[/_.-])(data|datasets?)([/_.-]|$)", re.IGNORECASE)
METRIC_WORDS = re.compile(r"acc|accuracy|error|err\b|loss|auc|asr|attack|success|dp\b|eopp|eodd|"
                          r"fairness|parity|f1|precision|recall|ap\b|mce|calib|fooling|similarity",
                          re.IGNORECASE)


# --------------------------------------------------------------------------- Dockerfile

def parse_dockerfile(text: str) -> dict:
    """WORKDIRs, ENV, where the code is put (git clone or COPY), and the base image."""
    logical, buf = [], ""
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.endswith("\\"):
            buf += s[:-1] + " "
            continue
        logical.append(buf + s)
        buf = ""
    if buf:
        logical.append(buf)

    info = {"base": "", "workdir": "/", "env": {}, "code_root": "", "code_from": "", "commit": "",
            "repo_url": "", "uses_copy": False, "evidence": [], "args": {}, "torch": ""}
    cwd = "/"
    for line in logical:
        op, _, rest = line.partition(" ")
        op = op.upper()
        if op == "FROM":
            info["base"] = rest.split()[0]
        elif op == "WORKDIR":
            cwd = posixpath.normpath(posixpath.join(cwd, rest.strip()))
            info["workdir"] = cwd
            info["evidence"].append(f"WORKDIR {cwd}")
        elif op == "ARG":
            name, _, default = rest.strip().partition("=")
            info["args"][name.strip()] = default.strip().strip("\"'")
        elif op == "ENV":
            for k, v in re.findall(r"""([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|'[^']*'|\S*)""", rest):
                info["env"][k] = v.strip("\"'")
        elif op == "RUN":
            tv = re.search(r"\btorch==([0-9]+\.[0-9]+(?:\.[0-9]+)?)", rest)
            if tv and not info["torch"]:
                info["torch"] = tv.group(1)
            m = re.search(r"git clone\s+(?:-[-\w=]+\s+)*(\S+)(?:\s+([^\s&;|]+))?", rest)
            if m:
                url, dest = m.group(1), m.group(2) or re.sub(r"\.git$", "", m.group(1).rstrip("/").split("/")[-1])
                info["repo_url"] = url
                info["code_root"] = posixpath.normpath(posixpath.join(cwd, dest))
                info["code_from"] = f"git clone {url} into {info['code_root']}"
                c = re.search(r"git checkout\s+([0-9a-fA-F]{7,40}|\S+)", rest)
                info["commit"] = c.group(1) if c else ""
        elif op in ("COPY", "ADD"):
            parts = [p for p in shlex.split(rest) if not p.startswith("--")]
            if len(parts) >= 2 and parts[0] in (".", "./"):
                info["uses_copy"] = True
                info["code_root"] = posixpath.normpath(posixpath.join(cwd, parts[-1]))
                info["code_from"] = f"COPY of the build context into {info['code_root']}"
    return info


def cuda_index_for(torch_version: str) -> str:
    """PyTorch's CUDA wheel index for a torch version (the newest CUDA it was built for)."""
    try:
        major, minor = (int(x) for x in torch_version.split(".")[:2])
    except ValueError:
        return ""
    table = {(1, 10): "cu113", (1, 11): "cu115", (1, 12): "cu116", (1, 13): "cu117", (2, 0): "cu118",
             (2, 1): "cu121", (2, 2): "cu121", (2, 3): "cu121", (2, 4): "cu124", (2, 5): "cu124",
             (2, 6): "cu126", (2, 7): "cu128", (2, 8): "cu128"}
    tag = table.get((major, minor), "cu128" if (major, minor) > (2, 8) else "")
    return f"https://download.pytorch.org/whl/{tag}" if tag else ""


# --------------------------------------------------------------------------- repo scanning

def _py_files(root: Path):
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            p = Path(folder) / name
            if name.endswith(".py") and p.stat().st_size < 1_000_000:
                yield p


def _parse(path: Path):
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, ValueError):
        return None


def documented_commands(root: Path, limit: int = 40) -> list:
    """Commands written in the repo's README / markdown / shell scripts."""
    out = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        if Path(folder).relative_to(root).parts.__len__() > 2:
            continue
        for name in files:
            if not name.lower().endswith((".md", ".sh", ".txt", ".rst")):
                continue
            path = Path(folder) / name
            text = path.read_text(encoding="utf-8", errors="ignore").replace("\\\n", " ")
            for i, line in enumerate(text.splitlines(), 1):
                s = re.sub(r"^\s*(\$|>)\s*", "", line).strip()
                if re.match(r"(python3?|bash|sh|torchrun|accelerate launch)\s+\S", s):
                    out.append({"command": s[:400], "where": f"{path.relative_to(root)}:{i}"})
                    if len(out) >= limit:
                        return out
    return out


def _own_argument_scan(root: Path) -> dict:
    found = {}
    for path in _py_files(root):
        tree = _parse(path)
        if tree is None:
            continue
        lines = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
                flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                kw = {k.arg: ast.unparse(k.value)[:80] for k in node.keywords if k.arg in ("default", "type", "choices", "help")}
                if flags:
                    lines.append(f"  {', '.join(flags)}  " + "  ".join(f"{k}={v}" for k, v in kw.items()))
        if lines:
            found[path.relative_to(root).as_posix()] = lines
    return found


def script_arguments(root: Path) -> tuple:
    """Command-line options of every script. Uses Team 3's list_arguments scanner when it is
    available (TEAM3_TOOLS_DIR pointing at a clone of team3-landseer-tools), else a local scan."""
    team3 = os.environ.get("TEAM3_TOOLS_DIR", "")
    if team3 and os.path.isdir(team3):
        try:
            if team3 not in sys.path:
                sys.path.insert(0, team3)
            from tools.list_arguments import scan_repo  # Team 3
            found, _ = scan_repo(root)
            return found, "Team 3 list_arguments"
        except Exception as e:  # their package failed to import here; fall back
            return _own_argument_scan(root), f"local scan (Team 3 scanner unavailable: {type(e).__name__})"
    return _own_argument_scan(root), "local scan (set TEAM3_TOOLS_DIR to use Team 3's scanner)"


def _local_imports(script: Path) -> list:
    """Modules the script imports from its own folder (where helper code like data loaders lives)."""
    tree = _parse(script)
    if tree is None:
        return []
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return [script.parent / f"{n}.py" for n in sorted(names) if (script.parent / f"{n}.py").is_file()]


def code_facts(files: list, root: Path) -> dict:
    """Hard-coded data paths, how results are printed, and run requirements, in the given files."""
    data_paths, outputs, wandb, cuda = [], [], False, False
    for path in files:
        tree = _parse(path)
        if tree is None:
            continue
        src_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        rel = path.relative_to(root).as_posix()
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                wandb = wandb or any(m and m.split(".")[0] == "wandb" for m in mods)
            if isinstance(node, ast.Attribute) and node.attr == "cuda":
                cuda = True
            literal = None
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literal = node.value
            elif (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "join"
                  and node.args and all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in node.args)):
                literal = "/".join(a.value for a in node.args)
            if (literal and "/" in literal and len(literal) < 200 and DATA_WORDS.search(literal)
                    and not re.match(r"https?://", literal) and " " not in literal):
                ln = getattr(node, "lineno", 0)
                data_paths.append({"path": literal, "where": f"{rel}:{ln}",
                                   "code": src_lines[ln - 1].strip()[:200] if ln else ""})
            if isinstance(node, ast.Call):
                fn = ast.unparse(node.func)
                if fn in ("print", "tabulate") or fn.endswith((".info", ".log", ".write", "json.dump")):
                    ln = node.lineno
                    text = src_lines[ln - 1].strip() if ln <= len(src_lines) else ""
                    if METRIC_WORDS.search(text) or fn == "tabulate":
                        in_loop, p = False, parents.get(node)
                        while p is not None:
                            if isinstance(p, (ast.For, ast.While)):
                                in_loop = True
                                break
                            p = parents.get(p)
                        outputs.append({"where": f"{rel}:{ln}", "code": text[:200],
                                        "inside_loop": in_loop})
    seen, unique = set(), []
    for d in data_paths:
        if d["path"] not in seen:
            seen.add(d["path"])
            unique.append(d)
    return {"data_paths": unique[:30], "outputs": outputs[:40], "uses_wandb": wandb, "calls_cuda": cuda}


# --------------------------------------------------------------------------- the two tool bodies

def find_run_commands(repo_path: str, dockerfile: str = "") -> dict:
    root = Path(os.path.expanduser(repo_path)).resolve()
    if not root.is_dir():
        return {"status": "ERROR", "message": f"repo folder not found: {root}"}
    args, scanner = script_arguments(root)
    entry = []
    for path in _py_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "__main__" in text and ("add_argument" in text or "sys.argv" in text):
            entry.append(path.relative_to(root).as_posix())
    result = {
        "status": "OK",
        "repo": str(root),
        "documented_commands": documented_commands(root),
        "entry_scripts": entry[:60],
        "script_options": {k: v[:40] for k, v in list(args.items())[:25]},
        "script_options_from": scanner,
    }
    if dockerfile:
        with open(os.path.expanduser(dockerfile)) as f:
            info = parse_dockerfile(f.read())
        result["container"] = {k: info[k] for k in ("base", "workdir", "code_root", "code_from", "commit", "env")}
    result["next_step"] = ("Pick the command that produces the result to reproduce (read the script with "
                           "read_file to confirm its options), then call generate_spec with it.")
    return result


def _host_path(container_path: str, code_root: str, root: Path):
    rel = posixpath.relpath(container_path, code_root) if code_root else None
    if not rel or rel.startswith(".."):
        return None
    p = root / rel
    return p if p.exists() else None


def _choose_mount(candidates: list, data_dir: Path, workdir: str):
    """Pick where data_dir must appear in the container: the candidate path whose tail exists
    inside data_dir. E.g. code reads ../datasets/adult/raw from /workspace/src, and data_dir holds
    adult/raw -> mount data_dir at /workspace/datasets."""
    for cand in candidates:
        full = posixpath.normpath(posixpath.join(workdir, cand["path"]))
        parts = [p for p in full.split("/") if p]
        for i in range(len(parts) + 1):
            tail = parts[i:]
            if tail and not (data_dir.joinpath(*tail)).exists():
                continue
            mount = "/" + "/".join(parts[:i])
            if mount != "/":
                return mount, cand, full
    return None, None, None


def generate_spec(papers_dir: str, paper_name: str, dockerfile: str, repo_path: str, command: str,
                  metrics: list, data_dir: str = "", data_mount: str = "", record_path: str = "",
                  gpu=None, timeout_minutes: int = 600, tolerance: float = 1.0, build_args=None) -> dict:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", paper_name).strip("_").lower() or "paper"
    root = Path(os.path.expanduser(repo_path)).resolve()
    df = Path(os.path.expanduser(dockerfile)).resolve()
    if not df.is_file():
        return {"status": "ERROR", "message": f"Dockerfile not found: {df}"}
    if not root.is_dir():
        return {"status": "ERROR", "message": f"repo folder not found: {root}"}
    if not metrics:
        return {"status": "ERROR", "message": "give at least one metric name"}

    info = parse_dockerfile(df.read_text())
    workdir = info["workdir"]
    evidence = [f"run from {workdir}: the Dockerfile's last WORKDIR"]
    if info["code_from"]:
        evidence.append(f"code: {info['code_from']}")
    warnings = []

    # which script the command runs, and the helper modules it imports from its own folder
    tokens = shlex.split(command)
    script_tokens = [t for t in tokens if t.endswith((".py", ".sh"))]
    files = []
    for script_token in script_tokens:
        cpath = posixpath.normpath(posixpath.join(workdir, script_token))
        host = _host_path(cpath, info["code_root"], root)
        if host is None:
            warnings.append(f"{script_token} does not resolve to a file in the repo from {workdir} "
                            f"(code is at {info['code_root'] or 'an unknown place'}) - check the command and workdir")
            continue
        found = [host] + (_local_imports(host) if host.suffix == ".py" else [])
        files += [f for f in found if f not in files]
        evidence.append(f"script: {host.relative_to(root)} (+ {len(found) - 1} helper module(s) it imports)")
    if not script_tokens:
        warnings.append("no .py or .sh script found in the command")
    facts = code_facts(files, root) if files else {"data_paths": [], "outputs": [], "uses_wandb": False,
                                                   "calls_cuda": False}

    # data: path-like command arguments, then paths hard-coded in the code
    arg_paths = []
    for i, t in enumerate(tokens):
        if t.startswith("-") and i + 1 < len(tokens) and re.search(r"data|dir|path|root", t, re.I) \
                and not tokens[i + 1].startswith("-"):
            arg_paths.append({"path": tokens[i + 1], "where": f"command option {t}"})
    candidates = facts["data_paths"] + arg_paths       # hard-coded paths first: they apply regardless of options
    volumes = {}
    if data_dir:
        ddir = Path(os.path.expanduser(data_dir)).resolve()
        if not ddir.is_dir():
            return {"status": "ERROR", "message": f"data folder not found: {ddir}"}
        if data_mount:
            mount, why = data_mount, "given explicitly"
        else:
            mount, cand, full = _choose_mount(candidates, ddir, workdir)
            why = f"code reads {full} ({cand['where']})" if mount else ""
        if mount:
            volumes[str(ddir)] = mount
            evidence.append(f"data: {ddir} mounted at {mount} because {why}")
        else:
            warnings.append(f"could not tell where the code reads its data; none of these paths ends in "
                            f"something inside {ddir}: {[c['path'] for c in candidates][:8]}. Pass data_mount.")
    elif candidates:
        evidence.append(f"no data folder given; the code refers to {[c['path'] for c in candidates][:5]} "
                        "(fine if it downloads its own data)")

    env = {}
    if facts["uses_wandb"] and "WANDB_MODE" not in info["env"]:
        env["WANDB_MODE"] = "disabled"
        evidence.append("WANDB_MODE=disabled: the code imports wandb, which otherwise asks for a login")
    use_gpu = facts["calls_cuda"] if gpu is None else bool(gpu)
    if use_gpu and re.search(r"--(device|gpu|cuda)[ =](cpu|-1|none)\b|--no[-_]cuda\b|--cpu\b", command, re.I):
        warnings.append("the command forces the CPU" + (", but the code calls .cuda() directly, so it will fail "
                        "or run far slower; remove the CPU option" if facts["calls_cuda"] else
                        "; remove that option to run on the GPU"))
    build_args = dict(build_args or {})
    index_default = info["args"].get("TORCH_INDEX_URL", "")
    if use_gpu and index_default.rstrip("/").endswith("/cpu") and "TORCH_INDEX_URL" not in build_args:
        cuda_index = cuda_index_for(info["torch"])
        if cuda_index:
            build_args["TORCH_INDEX_URL"] = cuda_index
            evidence.append(f"GPU build of torch {info['torch']}: the Dockerfile installs the CPU-only build by "
                            f"default, so TORCH_INDEX_URL is set to {cuda_index}")
        else:
            warnings.append("the Dockerfile installs the CPU-only build of torch and its version is unknown; "
                            "set TORCH_INDEX_URL to a CUDA index to use the GPU")

    record = {}
    if record_path:
        record = read_record(os.path.expanduser(record_path))
    spec_metrics = {}
    for metric in metrics:
        value = record.get(metric)
        spec_metrics[metric] = {
            "pattern": PATTERN_TODO,
            "occurrence": "last",
            "recorded": float(value) if value is not None else None,
            "tolerance": tolerance,
            "source": f"metrics record {os.path.basename(record_path)}" if value is not None else "",
        }
        if record_path and metric not in record:
            warnings.append(f"metric {metric!r} is not in the metrics record; its recorded value is empty")

    spec = {
        "name": name,
        "repo": info["repo_url"],
        "dockerfile": str(df),
        "context": str(root) if info["uses_copy"] else str(df.parent),
        "notes": ("Generated by generate_spec. Metric patterns are placeholders until the first run: "
                  "set them with update_spec_metric."),
    }
    if build_args:
        spec["build_args"] = build_args
    if record_path:
        spec["record"] = str(Path(os.path.expanduser(record_path)).resolve())
    spec["run"] = {"command": command, "workdir": workdir, "gpu": use_gpu, "timeout_minutes": timeout_minutes}
    if env:
        spec["run"]["env"] = env
    if volumes:
        spec["run"]["volumes"] = volumes
    out_dir = Path(papers_dir) / name
    old_path = out_dir / "spec.yaml"
    if old_path.exists():
        # Regenerating (e.g. a changed command) keeps what was already worked out per metric: the output
        # patterns, and the paper's values unless the record now supplies one.
        try:
            old_metrics = (yaml.safe_load(old_path.read_text()) or {}).get("metrics") or {}
        except yaml.YAMLError:
            old_metrics = {}
        kept = []
        for metric, fields in spec_metrics.items():
            prev = old_metrics.get(metric) or {}
            if prev.get("pattern") and prev["pattern"] != PATTERN_TODO:
                for k in ("pattern", "after", "occurrence", "scale"):
                    if k in prev:
                        fields[k] = prev[k]
                kept.append(metric)
            if fields["recorded"] is None and prev.get("recorded") is not None:
                fields["recorded"], fields["source"] = prev["recorded"], prev.get("source", "")
            if prev.get("tolerance") is not None and prev.get("tolerance") != tolerance:
                fields["tolerance"] = prev["tolerance"]
                if "tolerance_type" in prev:
                    fields["tolerance_type"] = prev["tolerance_type"]
        if kept:
            evidence.append(f"kept the output patterns already set for {', '.join(kept)} from the existing spec; "
                            "check them against the next run's matched_line")
    spec["metrics"] = spec_metrics
    if info["uses_copy"]:
        warnings.append(f"the Dockerfile copies the build context, so {root} must be a clone of the "
                        "paper's repo at the commit the paper uses")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "spec.yaml"
    text = yaml.safe_dump(spec, sort_keys=False, width=4096)
    path.write_text(text)
    load_spec(str(path))   # must load; raises otherwise
    return {
        "status": "OK",
        "spec_path": str(path),
        "spec": text,
        "how_it_was_decided": evidence,
        "warnings": warnings,
        "how_results_are_printed": facts["outputs"][:15],
        "next_steps": ("load_paper_spec, docker_build, then reproduce_metric. The first run reports each "
                       "metric as not found (placeholder pattern): find the result lines in the run log "
                       "(read_file with search), set pattern/occurrence with update_spec_metric, check the "
                       "matched_line it returns is the final result, then validate_metric."),
    }
