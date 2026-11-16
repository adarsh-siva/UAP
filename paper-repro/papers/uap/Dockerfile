FROM python:3.8-slim
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONPATH=/workspace:/workspace/uap
ENV PY=python
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url $TORCH_INDEX_URL --extra-index-url https://pypi.org/simple \
    torch==2.2.0 torchvision==0.17.0 && \
    pip install --no-cache-dir adversarial-robustness-toolbox==1.17.1 numpy==1.24.3 matplotlib==3.7.5 \
    Pillow==10.2.0 scikit-learn==1.3.2 tqdm==4.66.1 "setuptools<70"
WORKDIR /workspace/uap
RUN git clone https://github.com/adarsh-siva/UAP . && \
    git checkout b5dcb8b0d4b6634a1f79a600633fd5260b633f62
CMD ["bash"]
