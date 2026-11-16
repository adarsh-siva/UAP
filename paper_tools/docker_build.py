"""docker_build: build the paper's container image from its Dockerfile.

Uses Docker where it is installed, and Apptainer on clusters without Docker (such as Gilbreth);
see docker_ops.py. Pass the same ReproState instance to all reproduction tools in an agent.
"""
import json
import os
import time

from langchain_core.tools import tool

from . import docker_ops
from .state import ReproState

LOG_TAIL = 4000


def _tail(text: str, n: int = LOG_TAIL) -> str:
    return text if len(text) <= n else "...[truncated]...\n" + text[-n:]


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


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
    # One log per attempt, so an earlier failure is still readable after a rebuild.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = os.path.join(state.paper_dir("logs") if spec else state.workspace,
                       f"build-{tag.replace(':', '_').replace('/', '_')}-{stamp}.log")
    t0 = time.time()
    try:
        # build_args from the spec, e.g. TORCH_INDEX_URL for the CUDA build of PyTorch.
        rc, out = docker_ops.build(df, ctx, tag, log, build_args=spec.build_args if spec else None)
    except docker_ops.DockerError as e:
        return {"status": "ERROR", "message": str(e)}
    state.data["images"][tag] = {"dockerfile": df, "context": ctx, "ok": rc == 0, "log": log,
                                 "seconds": round(time.time() - t0, 1)}
    state.save()
    if rc != 0:
        return {"status": "FAILED", "image": tag, "log": log, "log_tail": _tail(out)}
    return {"status": "OK", "image": tag, "seconds": round(time.time() - t0, 1), "log": log}


def make_tools(state: ReproState) -> list:
    """Create docker_build bound to shared session state."""
    @tool
    def docker_build(dockerfile: str, tag: str = "", context: str = "") -> str:
        """Build a Docker image from a Dockerfile. Paths are relative to the active spec's folder.
        `tag` defaults to the spec's image tag; `context` defaults to the spec's build context.
        Returns status OK with the image tag, or FAILED with the tail of the build log."""
        return _dump(do_docker_build(state, dockerfile, tag, context))

    return [docker_build]
