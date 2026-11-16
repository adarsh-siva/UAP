"""All of Team 4's tools in one list, bound to one shared session state.

    from paper_tools.state import ReproState
    from paper_tools.toolkit import make_all_tools

    state = ReproState("workspace")
    tools = make_all_tools(state, papers_dir="papers")   # pass to the LangChain agent

Every tool returns an ERROR result instead of raising, so one failing call does not end the
agent's run.
"""
import json
import os

from . import (docker_build, read_tools, repo_tools, reproduce_tools, spec_gen_tools,
               spec_validation_tools, update_tools)
from .state import ReproState


def make_all_tools(state: ReproState, papers_dir: str) -> list:
    papers_dir = os.path.abspath(papers_dir)
    load_paper_spec, validate_metric = spec_validation_tools.make_tools(state)
    tools = [
        load_paper_spec,                                         # Aniketh
        *docker_build.make_tools(state),                         # Arjun
        *reproduce_tools.make_tools(state),                      # Tiffany
        validate_metric,                                         # Aniketh
        *update_tools.make_tools(state),                         # Samuel (update_spec_metric)
        *spec_gen_tools.make_tools(state, papers_dir),           # find_run_commands, generate_spec
        *repo_tools.make_tools(state, [state.workspace, papers_dir]),   # Felipe
        read_tools.read_file, read_tools.read_paper,             # Adarsh
    ]
    for t in tools:
        _return_errors(t)
    return tools


def _return_errors(t) -> None:
    """Make a tool hand any exception back to the agent as an ERROR result instead of crashing it."""
    func = t.func

    def safe(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            message = f"{type(e).__name__}: {e}"
            if "no paper spec loaded" in str(e):
                message += (". Create the paper's spec with generate_spec if it does not exist yet, "
                            "or call load_paper_spec with its path.")
            return json.dumps({"status": "ERROR", "message": message}, indent=2)

    t.func = safe
