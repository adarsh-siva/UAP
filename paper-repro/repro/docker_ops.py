"""Thin wrappers over the container runtime: Docker, or Apptainer on clusters without Docker."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Optional

OUTPUTS_MOUNT = "/repro_out"


class DockerError(RuntimeError):
    pass


def runtime() -> str:
    forced = os.environ.get("REPRO_RUNTIME")
    if forced:
        return forced
    if not shutil.which("docker") and shutil.which("apptainer"):
        return "apptainer"
    return "docker"


def images_dir() -> str:
    base = os.environ.get("REPRO_IMAGES_DIR") or os.path.join(os.environ.get("RCAC_SCRATCH", os.path.expanduser("~")), "repro", "images")
    os.makedirs(base, exist_ok=True)
    return base


def sandbox_path(tag: str) -> str:
    return os.path.join(images_dir(), re.sub(r"[^A-Za-z0-9_.-]+", "_", tag))


def dockerfile_to_def(text: str, build_args: Optional[dict] = None, context: str = "") -> str:
    """Translate a Dockerfile into an Apptainer definition file. COPY/ADD of local files become
    %files entries read from `context`, the build context folder."""
    build_args = build_args or {}
    files = []
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

    base, post, environment = None, [], []
    for line in logical:
        parts = line.split(None, 1)
        op, rest = parts[0].upper(), parts[1] if len(parts) > 1 else ""
        if op == "FROM":
            base = rest.split()[0]
        elif op == "ARG":
            name, _, default = rest.partition("=")
            post.append(f"export {name}={build_args.get(name, default)!r}")
        elif op == "ENV":
            pairs = re.findall(r"""([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|'[^']*'|\S*)""", rest)
            if not pairs and len(rest.split(None, 1)) == 2:
                pairs = [tuple(rest.split(None, 1))]
            for k, v in pairs:
                post.append(f"export {k}={v}")
                environment.append(f"export {k}={v}")
        elif op == "WORKDIR":
            post.append(f"mkdir -p {rest} && cd {rest}")
        elif op == "RUN":
            post.append(rest)
        elif op in ("COPY", "ADD"):
            files.extend(_copy_entries(op, rest, context, line))
    if not base:
        raise DockerError("Dockerfile has no FROM line")
    source_env = "[ -f /.singularity.d/env/10-docker2singularity.sh ] && . /.singularity.d/env/10-docker2singularity.sh || true"
    sections = ["Bootstrap: docker", f"From: {base}", ""]
    if files:
        # Apptainer copies %files before %post runs; COPY destinations rarely depend on earlier steps.
        sections += ["%files", *[f"{src} {dest}" for src, dest in files], ""]
    return "\n".join(sections + ["%post", source_env, *post, "", "%environment", *environment, ""])


def _copy_entries(op: str, rest: str, context: str, line: str) -> list:
    import json as _json
    import shlex
    rest = rest.strip()
    args = _json.loads(rest) if rest.startswith("[") else shlex.split(rest)
    if any(a.startswith("--from") for a in args):
        raise DockerError(f"multi-stage COPY --from is not supported for Apptainer: {line}")
    args = [a for a in args if not a.startswith("--")]          # --chown, --chmod
    if len(args) < 2:
        raise DockerError(f"cannot parse {op}: {line}")
    *srcs, dest = args
    if not context:
        raise DockerError(f"{op} needs a build context folder: {line}")
    entries = []
    for src in srcs:
        if re.match(r"https?://", src):
            raise DockerError(f"ADD from a URL is not supported for Apptainer: {line}")
        host = os.path.normpath(os.path.join(context, src))
        if not os.path.exists(host):
            raise DockerError(f"{op} source {src!r} not found in build context {context}")
        # "<dir>/." copies the folder's contents (hidden files included), like Docker's COPY <dir> <dest>.
        entries.append((host + "/." if os.path.isdir(host) else host, dest))
    return entries


def docker_bin() -> str:
    path = shutil.which("docker")
    if not path:
        raise DockerError(
            "docker CLI not found. Install Docker (see scripts/install_docker_wsl.sh) "
            "and make sure `docker info` works for this user.")
    return path


def _run(args: list, timeout: Optional[float], log_path: Optional[str] = None) -> tuple:
    """Run a command, streaming combined stdout/stderr to log_path. Returns (rc, output)."""
    if log_path:
        with open(log_path, "w") as log:
            try:
                proc = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                log.write(f"\n[repro] TIMEOUT after {timeout}s\n")
                rc = -1
        with open(log_path, errors="replace") as log:
            return rc, log.read()
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return -1, f"[repro] TIMEOUT after {timeout}s"


def build(dockerfile: str, context: str, tag: str, log_path: str, timeout: float = 3 * 3600,
          build_args: Optional[dict] = None) -> tuple:
    if runtime() == "apptainer":
        with open(dockerfile) as f:
            definition = dockerfile_to_def(f.read(), build_args, context)
        def_path = log_path + ".def"
        with open(def_path, "w") as f:
            f.write(definition)
        return _run(["apptainer", "build", "--fakeroot", "--force", "--sandbox", sandbox_path(tag), def_path],
                    timeout, log_path)
    args = [docker_bin(), "build", "-f", dockerfile, "-t", tag]
    for k, v in (build_args or {}).items():
        args += ["--build-arg", f"{k}={v}"]
    return _run(args + [context], timeout, log_path)


def _run_apptainer(sandbox: str, command: str, log_path: str, outputs_dir: str, gpu: bool, env: dict,
                   volumes: dict, workdir: str, timeout: Optional[float]) -> tuple:
    args = ["apptainer", "exec", "--writable", "--cleanenv", "--no-home"]
    if gpu:
        args.append("--nv")
    mounts = {outputs_dir: OUTPUTS_MOUNT, **(volumes or {})}
    for host, inside in mounts.items():
        os.makedirs(host, exist_ok=True)
        os.makedirs(sandbox + inside, exist_ok=True)
        args += ["-B", f"{host}:{inside}"]
    os.makedirs(sandbox + "/repro_home", exist_ok=True)
    merged = {"HOME": "/repro_home", **(env or {})}
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        merged.setdefault("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
    for k, v in merged.items():
        args += ["--env", f"{k}={v}"]
    if workdir:
        args += ["--pwd", workdir]
    return _run(args + [sandbox, "bash", "-c", command], timeout, log_path)


def is_running_container(name: str) -> bool:
    rc, out = _run([docker_bin(), "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"], 30)
    return rc == 0 and name in out.split()


def image_exists(tag: str) -> bool:
    rc, _ = _run([docker_bin(), "image", "inspect", tag], 30)
    return rc == 0


def run(container: str, command: str, *, log_path: str, outputs_dir: str, gpu: bool = True,
        env: Optional[dict] = None, volumes: Optional[dict] = None, workdir: str = "",
        shm_size: str = "8g", timeout: Optional[float] = None) -> tuple:
    """Run `command` in a running container (docker exec) or a fresh one from an image (docker run --rm)."""
    env = env or {}
    if runtime() == "apptainer":
        sandbox = container if os.path.isdir(container) else sandbox_path(container)
        if not os.path.isdir(sandbox):
            raise DockerError(f"no Apptainer image for '{container}' - run docker_build first")
        return _run_apptainer(sandbox, command, log_path, outputs_dir, gpu, env, volumes or {}, workdir, timeout)
    if is_running_container(container):
        args = [docker_bin(), "exec"]
        for k, v in env.items():
            args += ["-e", f"{k}={v}"]
        if workdir:
            args += ["-w", workdir]
        args += [container, "bash", "-lc", command]
    else:
        if not image_exists(container):
            raise DockerError(f"'{container}' is neither a running container nor a local image")
        args = [docker_bin(), "run", "--rm", "--shm-size", shm_size,
                "-v", f"{outputs_dir}:{OUTPUTS_MOUNT}"]
        if gpu:
            args += ["--gpus", "all"]
        for host, inside in (volumes or {}).items():
            args += ["-v", f"{host}:{inside}"]
        for k, v in env.items():
            args += ["-e", f"{k}={v}"]
        if workdir:
            args += ["-w", workdir]
        args += [container, "bash", "-lc", command]
    return _run(args, timeout, log_path)
