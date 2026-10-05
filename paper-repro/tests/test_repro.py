import json
import os

import pytest
from tabulate import tabulate

from repro import docker_ops
from repro.extract import compare, extract_value
from repro.spec import MetricSpec, load_spec
from repro.state import ReproState
from repro.tools import do_docker_build, do_load_spec, do_reproduce_metric, do_validate_metric, make_tools

PAPERS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "papers")

# Output in the exact formats each repo prints.
UAP_OUT = """[victim] test accuracy 0.8888 -> demo/checkpoint/victim.pth
FOOLING RATE =  0.7120
Best Fooling Rate =  0.7120
FOOLING RATE =  0.7895
Best Fooling Rate =  0.7895
ROC AUC (view level, 1000 vs 1000): 0.9698
ROC AUC (model level, 5 vs 5): 1.0000
"""

EP_OUT = """Before prunning
Training accuracy: 99.10
Validation accuracy: 93.20
Test clean accuracy: 93.51
Test attack success rate: 100.00
After EP prunning
Training accuracy: 97.00
Validation accuracy: 92.40
Test clean accuracy: 92.83
Test attack success rate: 0.92
After BNP prunning
Training accuracy: 96.10
Validation accuracy: 91.90
Test clean accuracy: 92.10
Test attack success rate: 1.37
After CLP prunning
Test clean accuracy: 91.20
"""

IPMIX_OUT = """Clean
\tTest Loss 0.155 | Test Error 4.35
Mean C Corruption Error: 8.912
"""


def ffb_out():
    headers = ["Step(Tr|Val|Te)", "acc", "ap", "dp", "eopp", "eodd"]
    rows = [[0, "70.00|69.00|68.00", "60.00|60.00|60.00", "5.00|5.00|5.00", "1.00|1.00|1.00", "2.00|2.00|2.00"],
            [140, "86.10|85.20|84.73", "77.10|76.20|75.30", "18.10|17.90|17.75", "8.00|8.10|8.20", "9.00|9.10|9.20"]]
    return "\x1b[2K" + tabulate(rows, headers=headers, tablefmt="grid", floatfmt="02.2f")


def metric(spec_dir, name):
    return load_spec(os.path.join(PAPERS, spec_dir, "spec.yaml")).metrics[name]


@pytest.mark.parametrize("spec_dir,name,text,expected", [
    ("uap", "model_auc", UAP_OUT, 1.0),
    ("uap", "view_auc", UAP_OUT, 0.9698),
    ("uap", "victim_accuracy", UAP_OUT, 88.88),
    ("uap", "uap_fooling_rate", UAP_OUT, 78.95),
    ("ep_bnp", "no_defense_acc", EP_OUT, 93.51),
    ("ep_bnp", "ep_asr", EP_OUT, 0.92),
    ("ep_bnp", "bnp_acc", EP_OUT, 92.10),
    ("ep_bnp", "bnp_asr", EP_OUT, 1.37),
    ("ipmix", "clean_error", IPMIX_OUT, 4.35),
    ("ipmix", "mce", IPMIX_OUT, 8.912),
    ("ffb", "test_acc", None, 84.73),
    ("ffb", "test_dp", None, 17.75),
])
def test_paper_patterns(spec_dir, name, text, expected):
    value = extract_value(text if text is not None else ffb_out(), metric(spec_dir, name))
    assert value == pytest.approx(expected)


def test_all_specs_load():
    for d in os.listdir(PAPERS):
        spec = load_spec(os.path.join(PAPERS, d, "spec.yaml"))
        assert spec.metrics and spec.run_command
        assert os.path.isfile(spec.resolve(spec.dockerfile))


def test_compare():
    m = MetricSpec("acc", "x", recorded=90.0, tolerance=1.0)
    assert compare(90.8, m)["status"] == "PASS"
    assert compare(88.5, m)["status"] == "FAIL"
    rel = MetricSpec("auc", "x", recorded=0.5, tolerance=0.1, tolerance_type="rel")
    assert compare(0.54, rel)["status"] == "PASS"
    assert compare(0.6, rel)["status"] == "FAIL"
    assert compare(1.0, MetricSpec("m", "x"))["status"] == "NO_RECORD"


def test_missing_pattern_returns_none():
    assert extract_value("nothing here", MetricSpec("m", r"acc ([0-9.]+)")) is None
    assert extract_value("acc 1", MetricSpec("m", r"acc ([0-9.]+)", after="NOPE")) is None


@pytest.fixture
def fake_docker(monkeypatch, tmp_path):
    calls = {"run": 0}

    def build(df, ctx, tag, log):
        open(log, "w").write("built")
        return 0, "built"

    def run(container, command, *, log_path, **kw):
        calls["run"] += 1
        open(log_path, "w").write(UAP_OUT)
        return 0, UAP_OUT

    monkeypatch.setattr(docker_ops, "build", build)
    monkeypatch.setattr(docker_ops, "run", run)
    return calls


def test_tool_flow_reuses_run(fake_docker, tmp_path):
    state = ReproState(str(tmp_path / "ws"))
    do_load_spec(state, os.path.join(PAPERS, "uap", "spec.yaml"))
    assert do_docker_build(state, "Dockerfile")["status"] == "OK"
    for name in ["model_auc", "view_auc", "victim_accuracy", "uap_fooling_rate"]:
        r = do_reproduce_metric(state, name, "")
        assert r["status"] == "OK"
        assert do_validate_metric(state, name)["status"] == "PASS"
    assert fake_docker["run"] == 1  # one container run served all four metrics
    # state persists across sessions
    again = ReproState(str(tmp_path / "ws"))
    assert again.data["validations"]["uap/model_auc"]["status"] == "PASS"


def test_validate_before_reproduce(tmp_path):
    state = ReproState(str(tmp_path / "ws"))
    do_load_spec(state, os.path.join(PAPERS, "uap", "spec.yaml"))
    assert do_validate_metric(state, "model_auc")["status"] == "ERROR"
    assert do_reproduce_metric(state, "nope", "")["status"] == "ERROR"


def test_langchain_tools(fake_docker, tmp_path):
    state = ReproState(str(tmp_path / "ws"))
    tools = {t.name: t for t in make_tools(state, [str(tmp_path)])}
    assert {"docker_build", "reproduce_metric", "validate_metric"} <= set(tools)
    tools["load_paper_spec"].invoke({"spec_path": os.path.join(PAPERS, "uap", "spec.yaml")})
    out = json.loads(tools["reproduce_metric"].invoke({"metric": "model_auc", "container": "repro-uap:latest"}))
    assert out["value"] == 1.0
    assert json.loads(tools["validate_metric"].invoke({"metric": "model_auc"}))["status"] == "PASS"
    assert "ERROR" in tools["write_file"].invoke({"path": "/etc/evil", "content": "x"})


PAPER_TEXT = """1 Introduction
Some text about BadNets.
5 Experiments
Table 1: Results compared with
other defense methods in CIFAR-10.
BadNets (A2O) BadNets (A2A)
Attacks ACC ASR ACC ASR
Origin 93.86 100.00 94.60 93.89
EP (Ours) 93.88 0.86 94.49 0.61
BNP (Ours) 93.60 1.60 94.25 0.72
Table 2: Tiny-ImageNet results.
"""


def test_read_paper_outline_span_and_nomatch(tmp_path):
    import re as _re
    state = ReproState(str(tmp_path / "ws"))
    rp = {t.name: t for t in make_tools(state, [str(tmp_path)])}["read_paper"]
    src = "/papers/fake.pdf"
    cache = tmp_path / "ws" / "paper_text"
    cache.mkdir(parents=True)
    (cache / (_re.sub(r"[^A-Za-z0-9]+", "_", src)[-80:] + ".txt")).write_text(PAPER_TEXT)

    outline = rp.invoke({"source": src})
    assert "3: Table 1: Results compared with" in outline and "10: Table 2" in outline
    span = rp.invoke({"source": src, "start_line": 3, "num_lines": 7})
    assert "8: EP (Ours) 93.88 0.86" in span
    nomatch = rp.invoke({"source": src, "search": r"ACC\s+\|"})
    assert "no '|' separators" in nomatch and "Table 1" in nomatch
    hit = rp.invoke({"source": src, "search": "Origin"})
    assert "EP (Ours)" in hit  # context window includes the neighbouring rows
    inside = rp.invoke({"source": src, "start_line": 6, "num_lines": 3})
    assert inside.startswith("3: Table 1") and "5: BadNets (A2O)" in inside  # backs up to the caption
    # search + start_line: the search still runs (run-5 bug), restricted to lines >= start_line
    both = rp.invoke({"source": src, "search": "Table 2", "start_line": 1})
    assert "10: Table 2" in both and "1: Some text" not in both
    late = rp.invoke({"source": src, "search": "Origin", "start_line": 9})
    assert late.startswith("no lines match")


def test_repeat_guard():
    from langchain_core.messages import ToolMessage
    from repro.agent import repeat_guard

    class Req:
        def __init__(self, args):
            self.tool_call = {"name": "read_paper", "args": args, "id": "c1"}

    guard = repeat_guard(max_repeats=2)
    runs = []

    def handler(req):
        runs.append(1)
        return ToolMessage(content="table rows", tool_call_id="c1", name="read_paper")

    first = guard.wrap_tool_call(Req({"start_line": 561}), handler)
    second = guard.wrap_tool_call(Req({"start_line": 561}), handler)
    third = guard.wrap_tool_call(Req({"start_line": 561}), handler)
    other = guard.wrap_tool_call(Req({"start_line": 558}), handler)
    assert first.content == "table rows"
    assert "already made this exact call" in second.content
    assert third.content.startswith("Refused") and len(runs) == 3   # third identical call not executed
    assert other.content == "table rows"


def test_update_metric_preserves_rest_of_spec(tmp_path):
    import shutil
    src_dir = tmp_path / "ep_bnp"
    shutil.copytree(os.path.join(PAPERS, "ep_bnp"), src_dir)
    spec_path = str(src_dir / "spec.yaml")
    import re as _re
    blank = _re.sub(r"recorded: [0-9.]+", "recorded: null", open(spec_path).read())
    open(spec_path, "w").write(blank)
    before = open(spec_path).read()
    state = ReproState(str(tmp_path / "ws"))
    tools = {t.name: t for t in make_tools(state, [str(tmp_path)])}
    tools["load_paper_spec"].invoke({"spec_path": spec_path})

    out = json.loads(tools["update_metric"].invoke(
        {"metric": "ep_acc", "recorded": 93.88, "source": "Table 1, BadNets (A2O), EP ACC"}))
    assert out["status"] == "OK"
    after = open(spec_path).read()
    spec = load_spec(spec_path)
    assert spec.metrics["ep_acc"].recorded == 93.88
    assert spec.metrics["ep_acc"].after == "After EP prunning"      # untouched fields survive
    assert spec.metrics["bnp_acc"].recorded is None                 # other metrics untouched
    changed = [l for l in after.splitlines() if l not in before.splitlines()]
    assert len(changed) == 2                                        # only recorded + source lines

    bad = json.loads(tools["update_metric"].invoke({"metric": "nope", "recorded": 1.0}))
    assert bad["status"] == "ERROR"

    # pattern edits are refused until there is real run output to test them on (run-4 failure mode)
    no_run = json.loads(tools["update_metric"].invoke(
        {"metric": "ep_acc", "pattern": r"ACC\s+(\d+\.\d+)", "after": "Origin"}))
    assert no_run["status"] == "ERROR" and "no successful run" in no_run["message"]
    assert load_spec(spec_path).metrics["ep_acc"].pattern == "Test clean accuracy: ([0-9.]+)"

    # ...but they never block the paper fields sent in the same call (GPU run-1 failure mode)
    mixed = json.loads(tools["update_metric"].invoke(
        {"metric": "bnp_acc", "recorded": 93.6, "source": "Table 1", "pattern": r"ACC\s+(\d+\.\d+)"}))
    assert mixed["status"] == "PARTIAL" and mixed["not_changed"] == ["pattern"]
    s = load_spec(spec_path)
    assert s.metrics["bnp_acc"].recorded == 93.6
    assert s.metrics["bnp_acc"].pattern == "Test clean accuracy: ([0-9.]+)"

    # with a run log, a pattern that matches the program's output is accepted, one that doesn't is refused
    log = tmp_path / "run.log"
    log.write_text(EP_OUT)
    spec = load_spec(spec_path)
    state.record_run("repro-ep_bnp:latest", spec.run_command, str(log), 0, 1.0)
    wrong = json.loads(tools["update_metric"].invoke({"metric": "ep_acc", "pattern": r"ACC\s+(\d+\.\d+)"}))
    assert wrong["status"] == "ERROR" and "nothing is found" in wrong["message"]
    right = json.loads(tools["update_metric"].invoke(
        {"metric": "ep_acc", "pattern": r"Test clean accuracy:\s*([0-9.]+)"}))
    assert right["status"] == "OK"


def test_write_file_rejects_broken_spec(tmp_path):
    state = ReproState(str(tmp_path / "ws"))
    wf = {t.name: t for t in make_tools(state, [str(tmp_path)])}["write_file"]
    target = tmp_path / "p" / "spec.yaml"
    target.parent.mkdir()
    target.write_text("name: ok\nrun: {command: echo}\nmetrics: {m: {pattern: 'x ([0-9]+)'}}\n")
    msg = wf.invoke({"path": str(target), "content": "name: x\nnotes: a: b: c\n  bad"})
    assert msg.startswith("ERROR")
    assert load_spec(str(target)).name == "ok"                      # previous version restored


@pytest.mark.parametrize("provider", ["ollama", "anthropic"])
def test_agent_builds(monkeypatch, tmp_path, provider):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from repro.agent import build_agent
    agent = build_agent(ReproState(str(tmp_path / "ws")), str(tmp_path / "papers"), provider=provider)
    assert agent is not None


def test_auto_provider(monkeypatch):
    from repro.agent import resolve_provider
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert resolve_provider("auto") == "ollama"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert resolve_provider("auto") == "anthropic"
