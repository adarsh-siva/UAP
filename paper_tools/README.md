# Paper tools

Team 4's tools for reproducing a paper's reported metrics: build the paper's container, run its
experiment, read each metric from the output, and compare it with the paper's value. Each tool is a
LangChain tool, and all of them share one `ReproState` (the active paper, runs, reproduced values,
verdicts).

## Tools

| Tool | File | Owner | What it does |
|---|---|---|---|
| `load_paper_spec(spec_path)` | `spec_validation_tools.py` | Aniketh | Load a paper's spec (YAML) and make it the active paper |
| `validate_metric(metric)` | `spec_validation_tools.py` | Aniketh | Compare the reproduced value with the paper's within the tolerance: PASS / FAIL / NO_RECORD |
| `docker_build(dockerfile, tag, context)` | `docker_build.py`, `docker_ops.py` | Arjun | Build the container image (Docker, or Apptainer where Docker is unavailable); one log per attempt |
| `reproduce_metric(metric, container)` | `reproduce_tools.py` | Tiffany | Run the experiment in the container and read the metric; returns the value and `matched_line`, the output line it came from |
| `update_spec_metric(metric, recorded, source, tolerance, pattern, after, occurrence)` | `update_tools.py` | Samuel | Edit one metric in the spec; changes to how output is read are tested on the real run log first |
| `read_paper(source, search, start_line, num_lines)` | `read_tools.py` | Adarsh | Read a paper PDF: outline, a span of lines, or a regex search |
| `read_file(path, start_line, max_lines, search)` | `read_tools.py` | Adarsh | Read numbered lines of a file, or search it with a regex |
| `inspect_repo(repo_url)` | `repo_tools.py` | Felipe | Clone a repository and summarize its files, README and dependency files |
| `write_file(path, content)` | `repo_tools.py` | Felipe | Write a file under the allowed folders; the paper's cloned code is read-only |
| `find_run_commands(repo_path, dockerfile)` | `spec_gen_tools.py`, `spec_gen.py` | Adarsh | Report how a repo can be run: documented commands, scripts and their options (Team 3's scanner), and the Dockerfile's layout |
| `generate_spec(paper_name, dockerfile, repo_path, command, metrics, ...)` | `spec_gen_tools.py`, `spec_gen.py` | Adarsh | Write the paper's spec from the chosen command: working folder, data mount, env, GPU and CUDA build, paper values from a Team 2 record |

Shared modules: `spec.py` (the spec format), `state.py` (session state), `extract.py` (reading a number
from output). `update_metric.py` at the repository root is Samuel's standalone JSON version of
Update_metric; `update_tools.py` is the version that works on the YAML spec and shared state.

## Use

```sh
python -m pip install -r requirements.txt
python -m pytest -q paper_tools/tests
```

All tools at once, for an agent:

```python
from paper_tools.state import ReproState
from paper_tools.toolkit import make_all_tools

state = ReproState("workspace")
tools = make_all_tools(state, papers_dir="papers")   # generate_spec writes papers/<paper>/spec.yaml
```

Or one person's tools, with the same state:

```python
from paper_tools import reproduce_tools, spec_validation_tools
load_paper_spec, validate_metric = spec_validation_tools.make_tools(state)
reproduce_metric, = reproduce_tools.make_tools(state)
```

`make_all_tools` also makes every tool return an ERROR result instead of raising, so one failed
call does not end an agent's run.

## The workflow

1. `inspect_repo` clones the code; `find_run_commands` and `read_file` show how it runs; the agent
   chooses the command (checking the paper's settings with `read_paper`).
2. `generate_spec` writes the spec. Metric patterns start as placeholders.
3. `docker_build` builds Team 1's Dockerfile.
4. `reproduce_metric` runs the experiment and saves the log. With a new spec it returns NOT_FOUND,
   because no pattern is set yet.
5. `update_spec_metric` sets each metric's pattern; the tool reads the saved log at once and returns
   the value and `matched_line`, to check it is the final result.
6. `read_paper` finds the paper's values; `update_spec_metric` records them with their source;
   `validate_metric` gives PASS or FAIL.

## The spec

One YAML file per paper:

```yaml
name: ffb
dockerfile: Dockerfile            # Team 1's Dockerfile
context: .                        # build context (needed when the Dockerfile uses COPY)
build_args: {TORCH_INDEX_URL: https://download.pytorch.org/whl/cu117}
record: metrics_record.json       # optional: Team 2's record with the paper's values
run:
  command: python -u ./ffb_tabular_erm.py --dataset adult ...
  workdir: /workspace/src
  gpu: true
  timeout_minutes: 600
  env: {WANDB_MODE: disabled}
  volumes: {/path/to/data/ffb: /workspace/datasets}   # host folder -> container path
metrics:
  acc:
    pattern: '140 \| \d+\.\d+\|\d+\.\d+\|(\d+\.\d+) \|'   # regex; the group is the number
    occurrence: last              # first, last, or an index
    after: null                   # optional: search only after this heading
    recorded: 85.35               # the paper's value
    source: Table 4, row Adult/Gender
    tolerance: 1.0
```

## Running on a cluster without Docker

When `docker` is not installed and `apptainer` is (as on Purdue's Gilbreth), `docker_ops.py`
converts the Dockerfile to an Apptainer definition (FROM, ARG, ENV, WORKDIR, RUN, and COPY/ADD
from the build context) and builds a sandbox with `apptainer build --fakeroot`. Runs use
`apptainer exec --nv` for the GPU, with the spec's mounts, env and working folder. Set
`REPRO_IMAGES_DIR` to choose where images are stored.

## Compatibility with the other teams

- **Team 1**: their Dockerfiles are used unchanged, including ones that `COPY` the repository.
- **Team 2**: a spec's `record:` field loads paper values from their metrics record (`{metric: value}`
  or `{"metrics": {...}}`); reproduced values are written to `reproduced_record.json` in the same format.
- **Team 3**: `find_run_commands` uses their `list_arguments` scanner when `TEAM3_TOOLS_DIR` points
  at a clone of `team3-landseer-tools`, and its own scan otherwise.
