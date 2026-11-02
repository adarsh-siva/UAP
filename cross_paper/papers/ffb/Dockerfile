# Candidate environment: build/import checks required before claiming compatibility.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN --mount=type=secret,id=pip_ca if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; pip install --no-cache-dir numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 scipy==1.14.1 statsmodels==0.14.4 tabulate==0.9.0 wandb==0.18.7
WORKDIR /workspace
RUN --mount=type=secret,id=pip_ca if [ -f /run/secrets/pip_ca ]; then export GIT_SSL_CAINFO=/run/secrets/pip_ca; fi; git clone --no-checkout https://github.com/Auspr7ng/fair_fairness_benchmark.git repository && cd repository && git checkout --detach 3efc8957cd63a62184de0f9452f2f51c3a254e53
WORKDIR /workspace/repository
ENV PYTHONUNBUFFERED=1 WANDB_MODE=offline
CMD ["python", "-c", "import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)"]
