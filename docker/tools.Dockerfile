FROM python:3.11-slim-bookworm
WORKDIR /app
COPY docker/requirements-tools.txt /app/docker/requirements-tools.txt
RUN --mount=type=secret,id=pip_ca \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install --no-cache-dir -r docker/requirements-tools.txt
COPY paper_tools /app/paper_tools
COPY update_metric.py test_update_metric.py /app/
COPY cross_paper /app/cross_paper
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "pytest", "-q", "paper_tools/tests", "cross_paper/tests", "test_update_metric.py"]
