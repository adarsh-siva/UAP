FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pandas scikit-learn scipy tabulate wandb statsmodels
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/ahxt/fair_fairness_benchmark.git
WORKDIR /workspace/fair_fairness_benchmark/src
ENV WANDB_MODE=offline
