# Local verification — November 2, 2026

Branch: `codex/docker-cross-paper-tests`, based on main `b5dcb8b`.
Draft review snapshot. Paper builds and GPU execution are not verified.

## Verified

- Docker Desktop 4.97.0 installed per-user using the official, signature-verified
  installer; Docker engine 29.9.2 is running with WSL 2 after a normal restart.
- Tools image `vip-paper-tools:test` built successfully (Python 3.11 Linux).
  Container run: **30 passed, 4 subtests passed in 4.24s**.
  Image manifest: `sha256:98738c934a78abbfab10b8a1280d2736654f5736cebc860e0f28cb63746ac0ed`.
- Initial build hit HTTPS CA verification failure. Supplying Windows-trusted
  public CA certificates via an optional BuildKit secret resolved it without
  disabling TLS verification or storing the bundle in the image.

- Windows Python 3.14 isolated environment: existing and new offline tests pass.
  Direct tool dependencies: langchain-core 1.7.0, PyYAML 6.0.3, pytest 9.1.1.
- Actual local clones of all four team forks contain the manifest-pinned commits
  and selected experiment entry points. These checks read Git blobs; they do not
  execute/import paper code or establish runtime compatibility.
- Four-paper fixture tests exercise the real loader, extractor and validator,
  including FFB `test/acc` and `test/dp` fields and distinct EP/BNP sections.
- Historical UAP log replay from
  `origin/paper-repro:paper-repro/test_runs/uap_full.run.log` extracts four values:

| Metric | Historical measurement | Inherited reference | Verdict |
| --- | ---: | ---: | --- |
| model_auc | 0.56 | 1.0 | FAIL |
| view_auc | 0.6143 | 0.9698 | FAIL |
| victim_accuracy (%) | 88.28 | 88.88 | PASS |
| uap_fooling_rate (%) | 82.90 | 78.95 | PASS |

This is replay of a previously committed log, not a newly executed experiment,
and is not the successful fork README run. Source SHA-256:
`2238a79be98744f91b42ecce97b9ad4c0e7a657072760ee530fc9a241f014a5e`.
Reference provenance is inherited and still needs independent review.

## Not verified / prerequisites

- Four paper images remain unverified. The first EP/BNP attempt downloaded the
  base and failed during pip installation with HTTPS certificate verification
  errors. Optional CA-secret support was subsequently added to the paper recipes
  and runner, but that build has not been rerun. This is separate from the
  verified tools image.
- Host GPU inventory reports Intel Graphics; NVIDIA GPU execution requires
  the teammate's machine or an appropriate allocated compute resource.
- Paper CUDA 12.8/PyTorch 2.8 environments are candidates adapted from the legacy
  files, not confirmations that old research code supports these versions.
- No fresh ML training/evaluation, GPU access check, external dataset preparation,
  checkpoint compatibility test, live LLM call or RCAC submission was performed.
- IPMix/FFB have no verified matching numeric references. NO_RECORD is expected
  when measurements become available; neither should be described as PASS.
- The unrelated existing `paper_tools/docker_build.py` issues are unchanged.

## Required before publishing as a verified Docker deliverable

1. On a Docker host, build/run the tools image and all four paper images.
2. Verify paper imports/CLI behavior, GPU access, and dataset/checkpoint paths.
3. Run selected experiments with approval for compute/time; preserve fresh reports.
4. Independently check reference tables, units, protocol equivalence and tolerances.
5. Update this status with actual outcomes before requesting merge review.
   Do not delete `paper-repro` as part of this work.
