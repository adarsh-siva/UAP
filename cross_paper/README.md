# Cross-paper reproduction checks

Local draft for team review. No claim of fresh paper reproduction or verified
Docker builds is made. The poster and full agent integration are outside this change.

## What this adds

- `docker/tools.Dockerfile`: CPU test image for loading specs, extracting values,
  validating metrics, and persisting results. It does not run a nested Docker daemon.
- `papers/*/Dockerfile`: candidate GPU experiment images for the four team forks,
  checked out at exact commits in `repositories.json`. Based on the legacy
  `paper-repro` Dockerfiles, with fork URLs and FFB paths updated.
- `check.py`: spec inspection, committed-repository checks, existing-output replay,
  and explicit Docker build/import/experiment modes using the existing team tools.
- Tests: four-paper output-format fixtures plus missing output, state persistence,
  CLI provenance, and missing Docker. Fixture values are not research results.

The tools image intentionally excludes intake/PDF/OCR/LLM dependencies because it
tests reproduction tools, not the standalone intake CLI. Paper images contain
PyTorch; the lightweight tools image does not. The tools requirements pin direct
dependencies; transitive packages and container base tags are not digest-locked.

## Experiment targets and prerequisites

| Paper | Selected protocol | Prerequisites |
| --- | --- | --- |
| UAP | Team fork full ResNet20 pipeline | NVIDIA GPU; CIFAR-10 download; long training |
| IPMix | CIFAR-10 WRN via `cifar.py` | NVIDIA GPU; CIFAR-10-C and mixing images under `data/ipmix`; CIFAR-10 download |
| EP/BNP | CIFAR-10 BadNets, default ResNet18, EP/BNP | NVIDIA GPU; CIFAR-10 download; poisoned-model training |
| FFB | Team fork ResNet20 ERM CelebA, Attractive/Male | `data/ffb/celeba/raw/celeba.csv`; 20k-row single-seed partial protocol |

These are individual-paper onboarding targets, not combined-defense experiments.
IPMix and EP/BNP retain the legacy supported protocols rather than claiming
ResNet20 runs without evidence. FFB deliberately replaces the legacy Adult
command because this team fork reorganized code under `ffb/` and supplies CelebA runners.
FFB/IPMix reference values are null where not verified. UAP/EP reference values
are inherited from `paper-repro` and still require independent table/protocol review.
Tolerances are inherited/provisional, not estimates from repeated runs.

## Local offline checks (repository root)

Use a separate Python environment, not the root legacy ML requirements:

```powershell
python -m venv .venv-cross-paper
.\.venv-cross-paper\Scripts\python.exe -m pip install -r docker/requirements-tools.txt
.\.venv-cross-paper\Scripts\python.exe -m pytest -q paper_tools/tests cross_paper/tests test_update_metric.py
.\.venv-cross-paper\Scripts\python.exe -m cross_paper.check ipmix --repo-path ../IPMix --output cross-paper-results/ipmix-inspect
```

Use `python` below for the activated environment. Each output directory must be
new to avoid cached results leaking between runs. Repository checks inspect
committed blobs, not working-tree edits; no research code is imported locally.

## Tools image

```powershell
docker build -f docker/tools.Dockerfile -t vip-paper-tools:test .
docker run --rm vip-paper-tools:test
```

On networks with a private HTTPS certificate authority, supply an approved PEM
CA bundle as a BuildKit secret; certificate verification stays enabled and the
bundle is not stored in the image. Normal networks need no secret:

```powershell
docker build --secret id=pip_ca,src=path/to/trusted-ca-bundle.pem -f docker/tools.Dockerfile -t vip-paper-tools:test .
```

Build and smoke-test a paper image on the Docker host (the runner records build
output, import output, image ID and source revision). Repeat for all four names:

```powershell
python -m cross_paper.check ipmix --build --smoke --output cross-paper-results/ipmix-build
```

If this host needs a private CA for HTTPS, add
`--ca-bundle path/to/trusted-ca-bundle.pem` to the build command. The bundle is
provided only during pip/Git build steps and is not baked into the image.

The default smoke test only imports torch/torchvision/numpy. It does not verify
CUDA access, paper execution, checkpoint compatibility or complete dependencies.
Image-specific CLI help checks can add evidence without training:

```powershell
docker run --rm vip-ffb:3efc8957cd63 python ffb/reproduce/reproduce_resnet20_celeba.py --help
```

## Existing result replay (no Docker, no training)

```powershell
python -m cross_paper.check uap --log path/to/actual-run.log --output cross-paper-results/uap-replay
python -m cross_paper.check ffb --log path/to/result.json --output cross-paper-results/ffb-replay
```

Only supply logs/results matching the chosen spec's experiment protocol.
Replay records the input hash and marks `new_experiment_executed=false`. It does
not independently establish log authenticity or paper-setting equivalence.
NOT_FOUND is an incomplete extraction, not a failed scientific comparison.
NO_RECORD means a measurement exists but a reference is missing. A completed
comparison may legitimately FAIL; workflow completion is separate from verdict.

## Actual experiments (explicit opt-in, potentially hours)

Populate data mounts first; review `papers/<name>/spec.yaml` before execution:

```powershell
python -m cross_paper.check ipmix --execute --output cross-paper-results/ipmix-experiment
```

Run this on a Docker host with an appropriate GPU and NVIDIA container support.
Missing data directories are refused. Execution uses `paper_tools/reproduce_tools.py`
and `docker_ops.py` directly; it does not use the currently broken LangChain
`docker_build.py` wrapper or change that teammate's implementation. Existing
runner limitations, including interrupted-subprocess cleanup, remain applicable.
No RCAC jobs are submitted. Gilbreth's Apptainer/Slurm runtime needs separate
integration; these Docker CLI commands are not claimed to work there unchanged.

## Review status

See `VERIFICATION.md` for observations and blockers. Generated reports, logs,
datasets, model weights, and credentials must stay out of the proposed commit.
