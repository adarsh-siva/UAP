#!/bin/bash
#SBATCH --account=zghodsi
#SBATCH --partition=a30
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
# Submit with: sbatch --job-name=<paper> --export=ALL,PAPER=<paper>,INSTRUCTIONS="..." scripts/gilbreth_agent.sbatch
BASE=$RCAC_SCRATCH/repro
cd "$BASE/paper-repro"
export APPTAINER_CACHEDIR=$BASE/apptainer_cache APPTAINER_TMPDIR=$BASE/tmp REPRO_RUNTIME=apptainer REPRO_IMAGES_DIR=$BASE/images
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
unset LD_PRELOAD
nvidia-smi -L
$BASE/miniforge/envs/repro/bin/python -u -m repro.cli --workspace "$BASE/workspace_$PAPER" agent --provider "${PROVIDER:-anthropic}" \
  --spec "papers/$PAPER/spec.yaml" --max-steps 120 --instructions "$INSTRUCTIONS"
echo "EXIT=$?"
