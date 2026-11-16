"""Repository tools for a LangChain agent (Team 4's Inspect_repo and Write_file).

inspect_repo clones a paper's code into the workspace and summarizes it; write_file writes
files the agent produces, only inside allowed folders, and never into the paper's code.
Pass the same ReproState instance to all reproduction tools in an agent.
"""
import json
import os
import re
import subprocess

from langchain_core.tools import tool

from .spec import load_spec
from .state import ReproState


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


def do_inspect_repo(state: ReproState, repo_url: str) -> dict:
    name = re.sub(r"\.git$", "", repo_url.rstrip("/").split("/")[-1])
    dest = os.path.join(state.workspace, "repos", name)
    if not os.path.isdir(dest):
        p = subprocess.run(["git", "clone", "-q", "--depth", "1", repo_url, dest],
                           capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            return {"status": "ERROR", "message": p.stderr[-2000:]}
    tree = []
    for root, dirs, files in os.walk(dest):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if os.path.relpath(root, dest).count(os.sep) > 2:
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
    return summary


def do_write_file(state: ReproState, roots: list, path: str, content: str) -> str:
    full = os.path.abspath(os.path.expanduser(path))
    if not any(full == r or full.startswith(r + os.sep) for r in roots):
        return f"ERROR: writes are only allowed under {roots}"
    repos = os.path.join(state.workspace, "repos")
    if full == repos or full.startswith(repos + os.sep):
        return ("ERROR: the paper's code is read-only. A reproduction must run the code as published; "
                "if it cannot run, report why instead of changing it.")
    os.makedirs(os.path.dirname(full), exist_ok=True)
    previous = None
    if os.path.exists(full):
        with open(full) as f:
            previous = f.read()
    with open(full, "w") as f:
        f.write(content)
    if os.path.basename(full) == "spec.yaml":
        # A spec written by hand must still load; otherwise put back what was there.
        try:
            load_spec(full)
        except Exception as e:
            if previous is None:
                os.remove(full)
            else:
                with open(full, "w") as f:
                    f.write(previous)
            return (f"ERROR: not saved - the spec does not load ({type(e).__name__}: {e}). "
                    "To change an existing spec, use update_spec_metric or generate_spec instead.")
    return f"wrote {len(content)} chars to {full}"


def make_tools(state: ReproState, writable_roots: list) -> list:
    """Create inspect_repo and write_file bound to shared session state. write_file may only
    write under `writable_roots` (e.g. the workspace and the papers folder)."""
    roots = [os.path.abspath(r) for r in writable_roots]

    @tool
    def inspect_repo(repo_url: str) -> str:
        """Clone a git repository (shallow) into the workspace and summarize it: file tree,
        README, and dependency files."""
        return _dump(do_inspect_repo(state, repo_url))

    @tool
    def write_file(path: str, content: str) -> str:
        """Write a text file (e.g. a helper script or notes). Only allowed under the workspace or
        papers folders, and never inside a cloned paper's code, which is read-only."""
        return do_write_file(state, roots, path, content)

    return [inspect_repo, write_file]
