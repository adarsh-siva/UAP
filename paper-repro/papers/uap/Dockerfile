FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir scikit-learn scipy tqdm pillow
ARG REF=main
WORKDIR /workspace
RUN git clone https://github.com/adarsh-siva/UAP.git && cd UAP && git checkout ${REF}
WORKDIR /workspace/UAP
ENV PY=python
