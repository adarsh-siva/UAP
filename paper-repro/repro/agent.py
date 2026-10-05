"""LangChain agent that drives the reproduction tools.

Two model backends:
  ollama     - a local model served by Ollama (default when ANTHROPIC_API_KEY is unset)
  anthropic  - Claude through the Anthropic API
"""
from __future__ import annotations

import json
import os

from langchain.agents import create_agent
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware, wrap_tool_call
from langchain_core.messages import ToolMessage

from .state import ReproState
from .tools import make_tools

ANTHROPIC_MODEL = "claude-opus-5-5"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_CTX = 32768

SYSTEM_PROMPT = """\
You reproduce the reported results of machine-learning papers from their public code, inside Docker.

The reproduction loop for a paper with a spec:
1. load_paper_spec(spec_path) - see the image, run command and metrics with their recorded values.
2. docker_build(dockerfile) - build the image. If the build fails, read the log tail, fix the
   Dockerfile with write_file, and rebuild.
3. reproduce_metric(metric, container) for each metric. Metrics that share a run command reuse the
   same run, so the first call may take hours and later ones return immediately.
   If a run fails or a pattern matches nothing, read the log tail, fix the cause (Dockerfile, run
   command, or the metric's pattern/after/occurrence in the spec), reload the spec, and retry.
4. validate_metric(metric) for each metric.

Onboarding a paper that has no spec yet:
- inspect_repo(repo_url), then read_file on the training/evaluation scripts to learn their
  command-line flags, data requirements, and the exact strings they print metrics with.
- To find the values the paper reports: call read_paper(pdf_url) first to get the outline of
  table captions, then read the relevant table with start_line. Check the code (read_file on the
  scripts' argument defaults) to see which experimental setting the run command actually runs,
  and pick the matching table, column and row. Record where each value came from in the metric's
  `source`. Never invent a recorded value; but before concluding a value is not in the paper,
  read every table whose caption could contain it.
- To change an existing spec, use update_metric (one metric at a time). Never rewrite an
  existing spec with write_file - use write_file only to create a new paper's files.
- write_file a Dockerfile and a spec.yaml under <papers_dir>/<paper_name>/, then run the loop.
- If the code needs data that cannot be downloaded automatically (e.g. Google Drive links,
  registration-gated datasets), stop and tell the user exactly which files to place in which host
  directory instead of guessing.

Spec format (YAML; relative paths resolve against the spec's folder):
  name: <short-id>
  repo: <git url>
  dockerfile: Dockerfile          # build file
  context: .                      # build context
  image: repro-<name>:latest      # optional tag
  notes: <free text: data requirements, deviations from the paper>
  run:
    command: <shell command run inside the container>
    workdir: /workspace/<repo>
    gpu: true
    timeout_minutes: 1440
    env: {KEY: value}
    volumes: {<host path>: <container path>}   # e.g. datasets; /repro_out is always mounted for outputs
  metrics:
    <metric_name>:
      pattern: '<regex with one capture group around the number>'
      after: '<optional regex; only search output after its last match>'
      occurrence: last            # first | last | index
      scale: 1.0
      file: <optional path under /repro_out to read instead of stdout>
      command: <optional per-metric command overriding run.command>
      recorded: <number reported by the paper, or null>
      tolerance: 0.5
      tolerance_type: abs         # abs | rel
      source: "Table 2, CIFAR-10, WRN-40-4"

Choose tolerances that reflect normal run-to-run variance for the metric (e.g. about one point of
accuracy for a single-seed CIFAR run) and state them in your final report.

Finish with a short report: for each metric the reproduced value, the recorded value, PASS/FAIL,
and any deviations from the paper's setup (fewer epochs, different hardware, missing data).
Report only what your tool results show. A metric is PASS or FAIL only if validate_metric
returned that verdict in this session; otherwise write "not reproduced" or "not validated".
Describe the spec's settings (GPU, data, commands) only as load_paper_spec reported them.
"""


def resolve_provider(provider: str = "auto") -> str:
    if provider != "auto":
        return provider
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "ollama"


def build_model(provider: str = "auto", model: str = "", fallbacks: bool = True, think: bool = True):
    provider = resolve_provider(provider)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model or OLLAMA_MODEL,
            base_url=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
            num_ctx=OLLAMA_CTX,
            temperature=0,
            reasoning=think,     # keep the model's thinking out of the message text
            keep_alive="30m",
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = {}
        if fallbacks:
            # Server-side refusal fallback: a declined request is re-run on a fallback model.
            kwargs = {"betas": ["server-side-fallback-2026-07-01"], "model_kwargs": {"fallbacks": "default"}}
        return ChatAnthropic(model=model or ANTHROPIC_MODEL, max_tokens=16000,
                             output_config={"effort": "high"}, **kwargs)
    raise ValueError(f"unknown provider '{provider}' (use ollama, anthropic, or auto)")


def repeat_guard(max_repeats: int = 2):
    """Stop the agent re-issuing an identical tool call (a common failure loop for small models)."""
    counts = {}

    @wrap_tool_call
    def guard(request, handler):
        call = request.tool_call
        key = (call["name"], json.dumps(call.get("args", {}), sort_keys=True))
        counts[key] = counts.get(key, 0) + 1
        n = counts[key]
        if n > max_repeats:
            return ToolMessage(
                content=(f"Refused: you have already called {call['name']} with exactly these arguments "
                         f"{n - 1} times and the result has not changed. Use the result you already have, "
                         "or take a different step (another tool, different arguments, or finish)."),
                tool_call_id=call["id"], name=call["name"])
        result = handler(request)
        if n > 1 and isinstance(result, ToolMessage):
            result = ToolMessage(
                content=f"{result.content}\n\n[note] You already made this exact call; the result is "
                        "unchanged. Act on it instead of repeating the call.",
                tool_call_id=result.tool_call_id, name=result.name)
        return result

    return guard


def build_agent(state: ReproState, papers_dir: str, provider: str = "auto", model: str = "",
                fallbacks: bool = True, think: bool = True):
    provider = resolve_provider(provider)
    tools = make_tools(state, writable_roots=[state.workspace, papers_dir])
    prompt = SYSTEM_PROMPT.replace("<papers_dir>", papers_dir)
    middleware = [repeat_guard()]
    if provider == "ollama":
        # Ollama silently drops the oldest tokens - including the system prompt - when the
        # conversation outgrows num_ctx. Clear old tool results well before that point.
        middleware.append(ContextEditingMiddleware(edits=[
            ClearToolUsesEdit(trigger=int(OLLAMA_CTX * 0.6), keep=3, exclude_tools=["load_paper_spec"]),
        ]))
    llm = build_model(provider, model, fallbacks, think)
    return create_agent(llm, tools, system_prompt=prompt, middleware=middleware)
