# Candidate environment: build/import checks required before claiming compatibility.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN --mount=type=secret,id=pip_ca if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; pip install --no-cache-dir numpy==1.26.4 pillow==11.0.0 scikit-learn==1.5.2
WORKDIR /workspace
RUN --mount=type=secret,id=pip_ca if [ -f /run/secrets/pip_ca ]; then export GIT_SSL_CAINFO=/run/secrets/pip_ca; fi; git clone --no-checkout https://github.com/huypham-0607/IPMix.git repository && cd repository && git checkout --detach c9f9a84cf3cb5cff3ceffd47e9942944f8691384
WORKDIR /workspace/repository
ENV PYTHONUNBUFFERED=1 WANDB_MODE=offline
CMD ["python", "-c", "import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)"]
