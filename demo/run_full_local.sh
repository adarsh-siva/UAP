#!/bin/bash
# Full-scale run on the local GPU: ResNet20 victim, paper-scale settings.
set -e
cd "$(dirname "$0")/.."
PY=$HOME/miniforge3/envs/uap/bin/python

export UAP_ARCH=resnet20
export UAP_VICTIM_EPOCHS=100
export UAP_INDEP_EPOCHS=100
export UAP_PIRACY_EPOCHS=10
export UAP_N_PIRACY=10
export UAP_N_INDEP=10
export UAP_POINTS=150
export UAP_VIEWS=200
export UAP_FIT_IMAGES=2000
export UAP_DELTA=0.17      # target 83% fooling rate
export UAP_ROUNDS=10

echo "=== step2: train model zoo (1 victim + 10 piracy + 10 indep, resnet20) ==="
$PY demo/step2_train_models.py
echo "=== step3: UAP generation ==="
$PY demo/step3_uap.py
echo "=== step4: fingerprint point selection (150x200) ==="
$PY demo/step4_fingerprint.py
echo "=== step5: encoder train/test ==="
$PY demo/step5_encoder.py
echo "=== step6: metrics ==="
$PY demo/step6_metrics.py
echo "=== FULL RUN DONE ==="
