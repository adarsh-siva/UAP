"""Spec-generation tools: find_run_commands and generate_spec.

find_run_commands reports how a cloned repo can be run (documented commands, entry scripts
and their options via Team 3's list_arguments scanner when TEAM3_TOOLS_DIR is set, and where
Team 1's Dockerfile puts and runs the code). generate_spec writes the paper's spec.yaml from
the chosen command: working folder, data mount, env, GPU and CUDA build, paper values from a
Team 2 metrics record. The logic is in spec_gen.py.
"""
import json
from typing import Optional

from langchain_core.tools import tool

from . import spec_gen
from .state import ReproState
from .update_tools import SPEC_LOCK


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


def make_tools(state: ReproState, papers_dir: str) -> list:
    """Create find_run_commands and generate_spec bound to shared session state.
    generate_spec writes papers_dir/<paper>/spec.yaml."""
    @tool
    def find_run_commands(repo_path: str, dockerfile: str = "") -> str:
        """Read a cloned repo (and optionally its Dockerfile) and report how it can be run:
        commands documented in its README/scripts, its entry scripts and their command-line options,
        and where the Dockerfile puts the code and runs it. Use it to choose the command for
        generate_spec."""
        return _dump(spec_gen.find_run_commands(repo_path, dockerfile))

    @tool
    def generate_spec(paper_name: str, dockerfile: str, repo_path: str, command: str, metrics: str,
                      data_dir: str = "", data_mount: str = "", record_path: str = "",
                      gpu: Optional[bool] = None, timeout_minutes: int = 600, tolerance: float = 1.0) -> str:
        """Write the paper's spec.yaml from its Dockerfile (Team 1), its cloned repo, the chosen run
        command and the metric names (comma-separated, as named in the metrics record).
        Works out the working folder, where to mount `data_dir` (from data paths in the command and
        paths hard-coded in the code), env settings such as WANDB_MODE, and GPU use; fills recorded
        values from `record_path` (Team 2's metrics record) when given. Metric patterns start as
        placeholders, set after the first run with update_spec_metric. Reports how each choice was
        made - check it - and loads the spec as the active paper."""
        names = [s.strip() for s in metrics.split(",") if s.strip()]
        with SPEC_LOCK:
            result = spec_gen.generate_spec(papers_dir, paper_name, dockerfile, repo_path, command, names,
                                            data_dir=data_dir, data_mount=data_mount, record_path=record_path,
                                            gpu=gpu, timeout_minutes=timeout_minutes, tolerance=tolerance)
        if result.get("status") == "OK":
            state.set_spec(result["spec_path"])
        return _dump(result)

    return [find_run_commands, generate_spec]
