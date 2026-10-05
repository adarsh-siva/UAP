# paper-repro

Reproduce the reported metrics of ML papers from their public code, inside Docker — either as a
fixed pipeline or driven by a LangChain agent (Claude Opus 5.5) that can also onboard new papers.

## The tools

| Tool | What it does |
|---|---|
| `docker_build(dockerfile)` | Builds the paper's image; returns the tag, or the failing build-log tail |
| `reproduce_metric(metric, container)` | Runs the paper's command in a running container (`docker exec`) or a fresh one from an image (`docker run --rm`, GPU + data volumes from the spec), and parses the metric from stdout or an output file. Metrics that share a command share one run. |
| `validate_metric(metric)` | Compares the reproduced value with the recorded value within the spec's tolerance → `PASS` / `FAIL` / `NO_RECORD` |

Supporting tools the agent uses to onboard a new paper: `load_paper_spec`, `inspect_repo`,
`read_file`, `write_file` (restricted to `workspace/` and `papers/`), `read_paper` (PDF text search,
to find the values the paper reports).

Nothing in the tools is paper-specific. Each paper is a folder under `papers/` with a `Dockerfile`
and a `spec.yaml`: repo, run command, data volumes, and per-metric regex, recorded value,
tolerance, and source. Metric extraction handles repeated labels (`after:` anchors to a section
header, `occurrence:` picks the match), ANSI-redrawn tables, scaling, and file-based outputs.

## Included papers

| Paper | Metrics | Recorded values | Data |
|---|---|---|---|
| `uap` — UAP fingerprinting (CVPR'22), ResNet-20 | model AUC, view AUC, victim accuracy, UAP fooling rate | filled in | auto-download |
| `ep_bnp` — Backdoor neurons (NeurIPS'22) | ACC / ASR before, after EP, after BNP | `null` — agent fills from paper | auto-download |
| `ipmix` — IPMix (NeurIPS'23) | clean error, mCE | `null` — agent fills from paper | CIFAR-10-C + mixing set → `data/ipmix/` |
| `ffb` — Fair Fairness Benchmark | ERM on Adult: test accuracy, DP gap | `null` — agent fills from paper | Dropbox zip → `data/ffb/` |

## Setup

```bash
sudo bash scripts/install_docker_wsl.sh        # Docker + NVIDIA container toolkit (WSL2 Ubuntu)
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...                    # for the agent only
python -m pytest -q tests                       # 19 offline tests, no Docker or API needed
```

## Model backends

The agent runs on a local model through Ollama by default (no API key needed), or on Claude when
`ANTHROPIC_API_KEY` is set. Choose explicitly with `--provider ollama|anthropic` and `--model`.

```bash
bash scripts/start_ollama.sh            # start the local server (needed after every reboot)
python -m repro.cli agent --provider ollama --model qwen3:8b --spec papers/ep_bnp/spec.yaml
```

Safeguards that hold whichever model is used: `update_metric` edits only the fields it is given;
pattern edits are refused unless they extract a number from a real run log; `write_file` refuses
to save a spec that does not load; repeated identical tool calls are cut off; and every run ends
with a verified summary read from saved state, which flags pass/fail claims the agent made up.

## Usage

```bash
# Fixed pipeline, no LLM: build -> reproduce every metric -> validate
python -m repro.cli run --spec papers/uap/spec.yaml

# Agent drives the same loop, fixing build/run/pattern problems as they come up
python -m repro.cli agent --spec papers/ep_bnp/spec.yaml \
    --instructions "Fill the recorded values from the paper before validating."

# Agent onboards a paper with no spec: reads the repo and paper, writes Dockerfile + spec, reproduces
python -m repro.cli agent --repo https://github.com/x/y --paper https://arxiv.org/pdf/XXXX.XXXXX \
    --metrics "CIFAR-10 clean accuracy for ResNet-18"

python -m repro.cli status                      # reproduced values and verdicts so far
```

State (built images, run logs, reproduced values, verdicts) persists in `workspace/state.json`,
so a long training run is never repeated just to read another metric.
