FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy pillow scikit-learn
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/hzlsaber/IPMix.git
WORKDIR /workspace/IPMix
