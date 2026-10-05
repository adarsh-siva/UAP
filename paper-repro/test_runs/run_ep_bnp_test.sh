#!/bin/bash
# Agent test: fill in the paper's reported values in the EP_BNP spec.
# Usage: bash test_runs/run_ep_bnp_test.sh <run-name>    (log goes to test_runs/<run-name>.log)
# Correct answers (Table 1, CIFAR-10, BadNets A2O): Origin 93.86/100.00, EP 93.88/0.86, BNP 93.60/1.60
cd "$(dirname "$0")/.."
NAME=${1:-run}
cp test_runs/ep_bnp_spec.blank.yaml papers/ep_bnp/spec.yaml
rm -rf workspace
SPEC=$PWD/papers/ep_bnp/spec.yaml
TASK="Load the paper spec $SPEC. Its metrics have no recorded values yet. The paper PDF is https://proceedings.neurips.cc/paper_files/paper/2022/file/76917808731dae9e6d62c2a7a6afb542-Paper-Conference.pdf and the code is https://github.com/RJ-T/NIPS2022_EP_BNP . Find the value the paper reports for each metric, matching the experimental setting that the spec's run command actually runs, and update the spec file so every metric has its recorded value and a source saying which table, column and row it came from. Keep everything else in the spec unchanged. Docker is not available on this machine, so do not build or run anything."
{
  ~/miniforge3/envs/repro/bin/python -u -m repro.cli agent --provider ollama --task "$TASK" --max-steps 60
  echo "EXIT=$?"
  cp papers/ep_bnp/spec.yaml "test_runs/$NAME.spec.yaml"
  ~/miniforge3/envs/repro/bin/python - <<'EOF'
from repro.spec import load_spec
key = {'no_defense_acc': 93.86, 'no_defense_asr': 100.0, 'ep_acc': 93.88,
       'ep_asr': 0.86, 'bnp_acc': 93.60, 'bnp_asr': 1.60}
try:
    s = load_spec('papers/ep_bnp/spec.yaml')
except Exception as e:
    print(f"GRADE: spec does not load ({e})")
    raise SystemExit
blank = load_spec('test_runs/ep_bnp_spec.blank.yaml')
right = sum(s.metrics[n].recorded == v for n, v in key.items())
patterns_kept = all((s.metrics[n].pattern, s.metrics[n].after) == (blank.metrics[n].pattern, blank.metrics[n].after) for n in key)
print(f"GRADE: {right}/6 values correct; output patterns {'unchanged' if patterns_kept else 'CHANGED'}")
EOF
} > "test_runs/$NAME.log" 2>&1
