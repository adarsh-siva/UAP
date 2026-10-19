# Paper tools

`spec_validation_tools.py` ports `load_paper_spec` and `validate_metric` from
`paper-repro/repro/tools.py` on the `paper-repro` branch. The shared `spec.py`,
`state.py`, and `extract.py` modules are copied from that branch without changes.
This contribution integrates the existing implementation into `main`.

Install the dependencies from the repository root:

```sh
python -m pip install langchain-core PyYAML pytest
python -m pytest -q paper_tools/tests/test_spec_validation_tools.py
```

Create the tools with one shared session state:

```python
from paper_tools.state import ReproState
from paper_tools.spec_validation_tools import make_tools

state = ReproState("workspace")
tools = make_tools(state)
# Append these two tools to the agent's other tools.
tools[0].invoke({"spec_path": "/path/to/paper/spec.yaml"})
```

`load_paper_spec` loads a YAML experiment configuration, persists the active paper,
and returns its settings and expected metrics. `validate_metric` compares a
previously reproduced measurement with the reference and tolerance, then persists
PASS, FAIL, or NO_RECORD. A missing measurement or unknown metric returns ERROR.
For example, 89.40 versus 88.88 with an absolute tolerance of 1.0 passes because
the difference is 0.52 percentage points.

Other reproduction tools must use the same `ReproState` instance and store measured
values in `state.data["reproduced"][state.key(metric)]` before validation. The
tests seed these values to exercise validation without Docker or model training.

This transfer does not include the agent, Docker execution tools, or paper
configurations. The remaining branch contents need separate integration before
the team removes `paper-repro`.
