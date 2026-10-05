cd "$(dirname "$0")/.."
for i in 1 2 3 4 5; do bash test_runs/run_ep_bnp_test.sh gpu_v2_run$i; done
echo done > test_runs/batch.done
