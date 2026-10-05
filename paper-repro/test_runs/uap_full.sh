cd /mnt/c/Users/adars/paper-repro
echo "start $(date)"
~/miniforge3/envs/repro/bin/python -u -m repro.cli run --spec papers/uap/spec.yaml
echo "EXIT=$? end $(date)"
