#!/bin/bash
#SBATCH --account=zghodsi
#SBATCH --partition=a30
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
# Onboarding run on Gilbreth. Two modes:
#   REPO set  -> --repo mode: the agent writes the Dockerfile and spec itself
#   TASK set  -> a free-form task (e.g. with a provided Dockerfile)
# Submit with: sbatch --export=ALL scripts/gilbreth_onboard.sbatch, after exporting
#   RUN_NAME and either REPO (+ PAPER_URL, METRICS, INSTRUCTIONS) or TASK.
BASE=$RCAC_SCRATCH/repro
cd "$BASE/paper-repro"
export APPTAINER_CACHEDIR=$BASE/apptainer_cache APPTAINER_TMPDIR=$BASE/tmp REPRO_RUNTIME=apptainer REPRO_IMAGES_DIR=$BASE/images
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$BASE/papers_$RUN_NAME"
unset LD_PRELOAD
export TEAM3_TOOLS_DIR=$BASE/team3-landseer-tools   # Team 3 argument scanner, used by find_run_commands
nvidia-smi -L
PY=$BASE/miniforge/envs/repro/bin/python
COMMON=(--workspace "$BASE/workspace_$RUN_NAME" agent --provider "${PROVIDER:-anthropic}" ${MODEL:+--model "$MODEL"} --papers-dir "$BASE/papers_$RUN_NAME" --max-steps 200)
if [ -n "$REPO" ]; then
  $PY -u -m repro.cli "${COMMON[@]}" --repo "$REPO" --paper "$PAPER_URL" --metrics "$METRICS" --instructions "$INSTRUCTIONS"
else
  $PY -u -m repro.cli "${COMMON[@]}" --task "$TASK"
fi
echo "EXIT=$?"
