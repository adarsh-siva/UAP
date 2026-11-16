"""Tests for spec generation (find_run_commands, generate_spec) and the updated tools."""
import json
import os

import pytest

from paper_tools import docker_ops
from paper_tools.extract import locate
from paper_tools.spec import MetricSpec, load_spec
from paper_tools.spec_gen import PATTERN_TODO, find_run_commands, generate_spec, parse_dockerfile
from paper_tools.state import ReproState
from paper_tools.toolkit import make_all_tools

# A small repo shaped like FFB: the script ignores --data_path and reads a hard-coded relative
# path, imports wandb, and reprints a results table inside its training loop.
SCRIPT = '''import argparse
import wandb
from dataset import load_adult_data
from tabulate import tabulate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../datasets/adult/raw")
    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--num_training_steps", type=int, default=150)
    args = parser.parse_args()
    X, y, s = load_adult_data(path="../datasets/adult/raw", sensitive_attribute="sex")
    for step in range(args.num_training_steps):
        if step % 10 == 0:
            print(tabulate(logs, headers=["Step", "acc", "dp"]))
'''
HELPER = '''import os
def load_adult_data(path, sensitive_attribute):
    return open(os.path.join(path, "adult.data")).read()
'''
README = """# FFB
Run:
```
python ffb_tabular_erm.py --dataset adult --sensitive_attr sex
```
"""
DOCKERFILE = """FROM python:3.8-slim
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y git
WORKDIR /workspace
RUN git clone https://github.com/ahxt/fair_fairness_benchmark . && \\
    git checkout 394f020e9d93cb195885ca5c67f411b175726e63
WORKDIR /workspace/src
CMD ["bash"]
"""
LOG = """| Step | acc | dp |
|    0 | 75.22|75.21|75.22 | 0.00|0.00|0.00 |
|  140 | 85.78|85.26|85.27 | 16.14|15.53|16.37 |
"""


@pytest.fixture
def ffb_like(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "ffb_tabular_erm.py").write_text(SCRIPT)
    (repo / "src" / "dataset.py").write_text(HELPER)
    (repo / "README.md").write_text(README)
    df = tmp_path / "team1" / "Dockerfile"
    df.parent.mkdir()
    df.write_text(DOCKERFILE)
    data = tmp_path / "data_ffb"
    (data / "adult" / "raw").mkdir(parents=True)
    (data / "adult" / "raw" / "adult.data").write_text("x")
    record = tmp_path / "metrics_record.json"
    record.write_text(json.dumps({"acc": 85.35, "dp": 16.67, "eodd": None}))
    return {"repo": repo, "dockerfile": df, "data": data, "record": record, "tmp": tmp_path}


def test_parse_dockerfile_finds_code_and_workdir():
    info = parse_dockerfile(DOCKERFILE)
    assert info["workdir"] == "/workspace/src"
    assert info["code_root"] == "/workspace"
    assert info["commit"].startswith("394f020")
    assert info["env"]["PYTHONUNBUFFERED"] == "1"


def test_find_run_commands(ffb_like):
    out = find_run_commands(str(ffb_like["repo"]), str(ffb_like["dockerfile"]))
    assert out["status"] == "OK"
    assert any("ffb_tabular_erm.py" in c["command"] for c in out["documented_commands"])
    assert "src/ffb_tabular_erm.py" in out["entry_scripts"]
    assert any("--data_path" in line for line in out["script_options"]["src/ffb_tabular_erm.py"])
    assert out["container"]["workdir"] == "/workspace/src"


def test_generate_spec_finds_hardcoded_data_path(ffb_like):
    papers = ffb_like["tmp"] / "papers"
    out = generate_spec(str(papers), "FFB adult", str(ffb_like["dockerfile"]), str(ffb_like["repo"]),
                        "python -u ffb_tabular_erm.py --dataset adult --data_path /ignored/by/the/code",
                        ["acc", "dp", "missing_metric"], data_dir=str(ffb_like["data"]),
                        record_path=str(ffb_like["record"]))
    assert out["status"] == "OK", out
    spec = load_spec(out["spec_path"])
    assert spec.workdir == "/workspace/src"
    # the code reads ../datasets/adult/raw from /workspace/src, and data_ffb holds adult/raw
    assert spec.volumes == {str(ffb_like["data"].resolve()): "/workspace/datasets"}
    assert spec.env == {"WANDB_MODE": "disabled"}
    assert spec.metrics["acc"].recorded == 85.35 and spec.metrics["dp"].recorded == 16.67
    assert spec.metrics["acc"].pattern == PATTERN_TODO and spec.metrics["acc"].occurrence == "last"
    assert spec.metrics["missing_metric"].recorded is None
    assert any("missing_metric" in w for w in out["warnings"])
    assert any(o["inside_loop"] for o in out["how_results_are_printed"])


def test_generated_spec_flow_with_matched_line(ffb_like, monkeypatch):
    def run(container, command, *, log_path, **kw):
        open(log_path, "w").write(LOG)
        return 0, LOG
    monkeypatch.setattr(docker_ops, "run", run)
    state = ReproState(str(ffb_like["tmp"] / "ws"))
    tools = {t.name: t for t in make_all_tools(state, str(ffb_like["tmp"] / "papers"))}
    gen = json.loads(tools["generate_spec"].invoke({
        "paper_name": "ffb", "dockerfile": str(ffb_like["dockerfile"]), "repo_path": str(ffb_like["repo"]),
        "command": "python -u ffb_tabular_erm.py --dataset adult", "metrics": "acc, dp",
        "data_dir": str(ffb_like["data"]), "record_path": str(ffb_like["record"])}))
    assert gen["status"] == "OK"

    first = json.loads(tools["reproduce_metric"].invoke({"metric": "acc"}))
    assert first["status"] == "NOT_FOUND" and "update_spec_metric" in first["message"]

    fix = json.loads(tools["update_spec_metric"].invoke(
        {"metric": "acc", "pattern": r"^\|\s*\d+\s*\|\s*[0-9.]+\|[0-9.]+\|([0-9.]+)", "occurrence": "last"}))
    assert fix["value_from_latest_run"] == 85.27
    assert fix["matched_line"].startswith("|  140 |")
    assert json.loads(tools["validate_metric"].invoke({"metric": "acc"}))["status"] == "PASS"

    record = json.load(open(os.path.join(state.paper_dir(), "reproduced_record.json")))
    assert record == {"acc": 85.27, "dp": None}          # Team 2's flat record format


def test_spec_reads_team2_record(tmp_path):
    (tmp_path / "rec.json").write_text(json.dumps({"metrics": {"Accuracy": 91.5}}))
    (tmp_path / "spec.yaml").write_text(
        "name: p\nrecord: rec.json\nrun: {command: echo}\nmetrics:\n"
        "  acc: {pattern: 'a ([0-9.]+)', record_key: Accuracy}\n"
        "  stated: {pattern: 'b ([0-9.]+)', recorded: 3.0}\n")
    spec = load_spec(str(tmp_path / "spec.yaml"))
    assert spec.metrics["acc"].recorded == 91.5 and "rec.json" in spec.metrics["acc"].source
    assert spec.metrics["stated"].recorded == 3.0


def test_locate_returns_line():
    value, line = locate(LOG, MetricSpec("acc", r"\|\s*\d+\s*\|\s*[0-9.]+\|[0-9.]+\|([0-9.]+)", occurrence="first"))
    assert value == 75.22 and line.startswith("|    0 |")


def test_copy_becomes_apptainer_files(tmp_path):
    ctx = tmp_path / "ipmix"
    ctx.mkdir()
    (ctx / "cifar.py").write_text("x")
    d = docker_ops.dockerfile_to_def("FROM python:3.10-slim\nWORKDIR /workspace\nCOPY . /workspace\n"
                                     "COPY cifar.py /opt/cifar.py\n", context=str(ctx))
    files = d.split("%files")[1].split("%post")[0]
    assert f"{ctx}/. /workspace" in files and f"{ctx}/cifar.py /opt/cifar.py" in files
    with pytest.raises(docker_ops.DockerError):
        docker_ops.dockerfile_to_def("FROM x\nCOPY missing.txt /a\n", context=str(ctx))
    with pytest.raises(docker_ops.DockerError):
        docker_ops.dockerfile_to_def("FROM x\nCOPY --from=build /a /b\n", context=str(ctx))


def test_write_file_refuses_paper_code(tmp_path):
    state = ReproState(str(tmp_path / "ws"))
    wf = {t.name: t for t in make_all_tools(state, str(tmp_path / "papers"))}["write_file"]
    target = os.path.join(state.workspace, "repos", "ffb", "src", "dataset.py")
    assert "read-only" in wf.invoke({"path": target, "content": "x"})
    assert not os.path.exists(target)


def test_generate_spec_gpu_torch_for_team1_dockerfiles(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("import torch\nx = torch.zeros(1).cuda()\n")
    (repo / "test.py").write_text("from defense import EP\n")
    (repo / "defense.py").write_text("def EP(m):\n    return m.cuda()\n")
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.8-slim\nARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu\n"
                  "RUN pip install --index-url $TORCH_INDEX_URL torch==1.12.1 torchvision==0.13.1\n"
                  "WORKDIR /workspace\nRUN git clone https://github.com/x/y . && git checkout abc1234\n")
    out = generate_spec(str(tmp_path / "papers"), "ep", str(df), str(repo),
                        "python train.py --device cpu && python test.py --device cpu", ["acc"])
    spec = load_spec(out["spec_path"])
    assert spec.gpu is True
    assert spec.build_args == {"TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu116"}
    assert any("forces the CPU" in w for w in out["warnings"])
    assert sum("script:" in e for e in out["how_it_was_decided"]) == 2      # both scripts were read


def test_regenerating_spec_keeps_patterns_and_recorded(ffb_like):
    papers = str(ffb_like["tmp"] / "papers")
    args = (papers, "ffb", str(ffb_like["dockerfile"]), str(ffb_like["repo"]))
    first = generate_spec(*args, "python -u ffb_tabular_erm.py --dataset adult", ["acc", "dp"])
    import yaml
    data = yaml.safe_load(open(first["spec_path"]))
    data["metrics"]["acc"].update(pattern=r"acc ([0-9.]+)", recorded=85.35, source="Table 4")
    yaml.safe_dump(data, open(first["spec_path"], "w"), sort_keys=False)
    second = generate_spec(*args, "python -u ffb_tabular_erm.py --dataset adult --batch_size 128", ["acc", "dp"])
    spec = load_spec(second["spec_path"])
    assert "--batch_size 128" in spec.run_command
    assert spec.metrics["acc"].pattern == r"acc ([0-9.]+)" and spec.metrics["acc"].recorded == 85.35
    assert spec.metrics["dp"].pattern == PATTERN_TODO
    assert any("kept the output patterns" in e for e in second["how_it_was_decided"])


def test_state_save_is_thread_safe(tmp_path):
    import threading
    state = ReproState(str(tmp_path / "ws"))
    errors = []

    def work(i):
        try:
            for _ in range(50):
                state.data["runs"][str(i)] = i
                state.save()
        except Exception as e:     # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert not [p for p in os.listdir(state.workspace) if p.endswith(".tmp")]


EP_SPEC = """name: ep_bnp
repo: https://github.com/RJ-T/NIPS2022_EP_BNP
run:
  command: python train.py && python test.py
  gpu: false
metrics:
  ep_acc:
    pattern: 'Test clean accuracy: ([0-9.]+)'
    after: After EP prunning
    recorded: null
  bnp_acc:
    pattern: 'Test clean accuracy: ([0-9.]+)'
    after: After BNP prunning
    recorded: null
"""
EP_OUT = """Before prunning
Test clean accuracy: 93.51
After EP prunning
Test clean accuracy: 93.30
After BNP prunning
Test clean accuracy: 92.10
"""


def test_dockerfile_to_apptainer_def():
    df = ("FROM python:3.8-slim\nARG TORCH_INDEX_URL=https://example/cpu\nENV WANDB_MODE=disabled\n"
          "WORKDIR /workspace/src\nRUN pip install --index-url $TORCH_INDEX_URL torch\nCMD [\"bash\"]\n")
    d = docker_ops.dockerfile_to_def(df, {"TORCH_INDEX_URL": "https://example/gpu"})
    assert "From: python:3.8-slim" in d
    assert "export TORCH_INDEX_URL='https://example/gpu'" in d
    assert d.index("cd /workspace/src") < d.index("pip install")
    assert d.split("%environment")[1].strip() == "export WANDB_MODE=disabled"


def test_update_spec_metric_keeps_rest_and_checks_patterns(tmp_path):
    spec_path = tmp_path / "ep_bnp" / "spec.yaml"
    spec_path.parent.mkdir()
    spec_path.write_text(EP_SPEC)
    before = spec_path.read_text()
    state = ReproState(str(tmp_path / "ws"))
    tools = {t.name: t for t in make_all_tools(state, str(tmp_path))}
    tools["load_paper_spec"].invoke({"spec_path": str(spec_path)})

    out = json.loads(tools["update_spec_metric"].invoke(
        {"metric": "ep_acc", "recorded": 93.88, "source": "Table 1, EP ACC"}))
    assert out["status"] == "OK"
    spec = load_spec(str(spec_path))
    assert spec.metrics["ep_acc"].recorded == 93.88 and spec.metrics["ep_acc"].after == "After EP prunning"
    assert spec.metrics["bnp_acc"].recorded is None
    changed = [l for l in spec_path.read_text().splitlines() if l not in before.splitlines()]
    assert len(changed) == 2                                   # only recorded + source

    # reading rules are refused until there is a real run to test them on...
    no_run = json.loads(tools["update_spec_metric"].invoke({"metric": "ep_acc", "pattern": r"ACC (\d+)"}))
    assert no_run["status"] == "ERROR" and "no successful run" in no_run["message"]
    # ...but never block the paper fields sent with them
    mixed = json.loads(tools["update_spec_metric"].invoke(
        {"metric": "bnp_acc", "recorded": 93.6, "source": "Table 1", "pattern": r"ACC (\d+)"}))
    assert mixed["status"] == "PARTIAL" and load_spec(str(spec_path)).metrics["bnp_acc"].recorded == 93.6

    log = tmp_path / "run.log"
    log.write_text(EP_OUT)
    state.record_run(state.spec.image_tag(), state.spec.run_command, str(log), 0, 1.0)
    wrong = json.loads(tools["update_spec_metric"].invoke({"metric": "ep_acc", "pattern": r"ACC (\d+)"}))
    assert wrong["status"] == "ERROR" and "nothing is found" in wrong["message"]
    right = json.loads(tools["update_spec_metric"].invoke(
        {"metric": "ep_acc", "pattern": r"Test clean accuracy:\s*([0-9.]+)", "occurrence": "first"}))
    assert right["value_from_latest_run"] == 93.30 and right["matched_line"] == "Test clean accuracy: 93.30"
    assert json.loads(tools["update_spec_metric"].invoke(
        {"metric": "ep_acc", "occurrence": "middle"}))["status"] == "ERROR"


def test_write_file_rejects_broken_spec(tmp_path):
    state = ReproState(str(tmp_path / "ws"))
    wf = {t.name: t for t in make_all_tools(state, str(tmp_path))}["write_file"]
    target = tmp_path / "p" / "spec.yaml"
    target.parent.mkdir()
    target.write_text("name: ok\nrun: {command: echo}\nmetrics: {m: {pattern: 'x ([0-9]+)'}}\n")
    assert wf.invoke({"path": str(target), "content": "name: x\nnotes: a: b: c\n  bad"}).startswith("ERROR")
    assert load_spec(str(target)).name == "ok"                 # previous version restored


def test_tool_errors_are_returned_not_raised(tmp_path):
    state = ReproState(str(tmp_path / "ws"))
    tools = {t.name: t for t in make_all_tools(state, str(tmp_path))}
    for name, args in [("reproduce_metric", {"metric": "acc"}), ("validate_metric", {"metric": "acc"}),
                       ("update_spec_metric", {"metric": "acc", "recorded": 1.0})]:
        out = json.loads(tools[name].invoke(args))
        assert out["status"] == "ERROR" and "load_paper_spec" in out["message"]


def test_read_file_search_and_limits(tmp_path):
    from paper_tools.read_tools import read_file as rf
    f = tmp_path / "train.py"
    f.write_text("import os\n" + "x = 1\n" * 50 + "def metric_evaluation(y):\n    return y\n" + "\x1b[2K| 140 | 85.2 |\n")
    hit = rf.invoke({"path": str(f), "search": "def metric_evaluation"})
    assert "52\tdef metric_evaluation(y):" in hit and "1 matching lines" in hit
    assert "\x1b" not in rf.invoke({"path": str(f), "start_line": 50})
    big = tmp_path / "big.log"
    big.write_text(("y" * 300 + "\n") * 300)
    assert len(rf.invoke({"path": str(big)})) < 13000


def test_tail_drops_apptainer_warnings():
    from paper_tools.reproduce_tools import _tail
    out = _tail("WARNING: Skipping mount /depot [binds]\nTraceback\nFileNotFoundError: x")
    assert "Skipping mount" not in out and "FileNotFoundError" in out
