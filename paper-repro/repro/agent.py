"""LangChain agent that drives the reproduction tools.

Model backends:
  anthropic  - Claude through the Anthropic API (default when an Anthropic API key is set)
  rcac       - Purdue RCAC GenAI Studio (OpenAI-compatible)
  ollama     - a local model served by Ollama (used when no API key is set)
"""
from __future__ import annotations

import json
import os

from langchain.agents import create_agent
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware, wrap_model_call, wrap_tool_call
from langchain_core.messages import ToolMessage

from .state import ReproState
from .tools import make_tools

ANTHROPIC_MODEL = "claude-opus-5-5"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_CTX = 32768
RCAC_BASE_URL = "https://genai.rcac.purdue.edu/api"
RCAC_MODEL = "qwen3.8:27b"
# RCAC models (qwen3.8:27b, gpt-oss:120b) accept at most 65,536 tokens; ~3 chars per token for code and logs
RCAC_CONTEXT_CHARS = 110_000
RCAC_KEY_FILE = os.path.expanduser("~/.config/rcac_genai_key")
ANTHROPIC_KEY_FILE = os.path.expanduser("~/.config/anthropic_api_key")

SYSTEM_PROMPT = """\
You reproduce the reported results of machine-learning papers from their public code, inside
containers. Other teams supply two inputs: a Dockerfile for the paper's code (Team 1) and, often,
a metrics record with the values the paper reports (Team 2). Your job is everything between:
work out how to run the code, write the spec, build, run, read the results, and validate them.

Steps for a paper without a spec:
1. inspect_repo(repo_url) to clone the code, then find_run_commands(repo_path, dockerfile) to see
   the commands the repo documents, its scripts and their options, and where the Dockerfile puts
   and runs the code. Read the relevant script with read_file (use `search` to jump to argument
   definitions, data loading and print statements) to choose the exact command for the result
   you are asked to reproduce. Prefer the repo's documented command with the options the paper's
   setting needs; keep the script's defaults unless the paper's setting differs.
   The experiment runs on a GPU: do not add CPU-only options (such as --device cpu).
   Change the command only to match settings the paper or README states (batch size, epochs, ...).
   Never change the random seed, or rerun hoping for a different number, to make a metric pass:
   a FAIL after matching the stated settings is a valid result - report it.
2. generate_spec(...) with that command, the metric names, the data folder if one was given, and
   the metrics record if one was given. Read its `how_it_was_decided` and `warnings`, and check
   them against the code: especially the working folder and where the data is mounted. If a
   choice is wrong, call generate_spec again with corrected arguments (e.g. data_mount).
3. docker_build(dockerfile). The Dockerfile belongs to another team: do not rewrite it. If it
   does not build, report the build error.
4. reproduce_metric(metric) for the first metric. This runs the experiment (it may take hours).
   With a new spec the result is NOT_FOUND, because metric patterns start as placeholders: find
   the line(s) holding the final result in the run log (read_file on the log path with `search`),
   then update_spec_metric(metric, pattern=..., occurrence=...). Its reply shows `matched_line`,
   the exact output line the number was read from - confirm it is the FINAL result (last epoch or
   step, test split, the right column and metric), not an earlier step or another column.
   The pattern is a Python regex with one capture group around the number. Logs often reprint
   a table every few steps, so the final value is usually occurrence "last".
5. reproduce_metric for each remaining metric (the finished run is reused), fixing patterns the
   same way, then validate_metric(metric) for every metric.

If a run fails (RUN_FAILED), read the log tail and fix what is yours to fix: the command, working
folder, data mount or env in the spec (generate_spec again). Never change the paper's code - it is
read-only - and never edit the Dockerfile. If the code cannot run as published, say why.

Recorded values: when a metrics record is given they come from it. Otherwise find them in the
paper with read_paper (outline first, then the table with start_line), matching the setting your
command runs, and save each with update_spec_metric(metric, recorded=..., source="Table N, row,
column"). Never invent a value.

If the code needs data that is not on this machine and cannot be downloaded automatically, stop
and say exactly which files are needed and where.

Finish with a short report: for each metric the reproduced value, the matched output line, the
recorded value, and PASS/FAIL, plus any deviations from the paper's setup. Report only what your
tool results show: a metric is PASS or FAIL only if validate_metric returned that verdict.
"""


def rcac_key() -> str:
    key = os.environ.get("RCAC_GENAI_API_KEY", "")
    if not key and os.path.exists(RCAC_KEY_FILE):
        with open(RCAC_KEY_FILE) as f:
            key = f.read().strip()
    return key


def anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key and os.path.exists(ANTHROPIC_KEY_FILE):
        with open(ANTHROPIC_KEY_FILE) as f:
            key = f.read().strip()
    return key


def resolve_provider(provider: str = "auto") -> str:
    if provider != "auto":
        return provider
    if anthropic_key():
        return "anthropic"
    return "rcac" if rcac_key() else "ollama"


def build_model(provider: str = "auto", model: str = "", fallbacks: bool = True, think: bool = True):
    provider = resolve_provider(provider)
    if provider == "rcac":
        from langchain_openai import ChatOpenAI
        key = rcac_key()
        name = model or os.environ.get("RCAC_GENAI_MODEL", RCAC_MODEL)
        if not key:
            raise ValueError(f"set RCAC_GENAI_API_KEY or put the key in {RCAC_KEY_FILE}")
        return ChatOpenAI(model=name, base_url=RCAC_BASE_URL, api_key=key, temperature=0, timeout=900, max_retries=4)
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
        key = anthropic_key()
        if not key:
            raise ValueError(f"set ANTHROPIC_API_KEY or put the key in {ANTHROPIC_KEY_FILE}")
        return ChatAnthropic(model=model or ANTHROPIC_MODEL, api_key=key, max_tokens=16000,
                             output_config={"effort": "high"}, max_retries=4, **kwargs)
    raise ValueError(f"unknown provider '{provider}' (use ollama, anthropic, or auto)")


# Tools that change the spec, the image or the runs: after one of them, earlier results may no longer hold.
STATE_CHANGING_TOOLS = {"update_spec_metric", "generate_spec", "docker_build", "reproduce_metric",
                        "load_paper_spec", "write_file"}


def repeat_guard(max_repeats: int = 2):
    """Stop the agent re-issuing an identical tool call (a common failure loop for small models).
    Counts reset whenever a state-changing tool runs, since the same call can then give a new result."""
    counts, last_result = {}, {}
    generation = [0]

    @wrap_tool_call
    def guard(request, handler):
        call = request.tool_call
        key = (generation[0], call["name"], json.dumps(call.get("args", {}), sort_keys=True))
        counts[key] = counts.get(key, 0) + 1
        n = counts[key]
        if n > max_repeats:
            return ToolMessage(
                content=(f"Refused: you have already called {call['name']} with exactly these arguments "
                         f"{n - 1} times and nothing has changed since. Use the result you already have, "
                         "or take a different step (another tool, different arguments, or finish)."),
                tool_call_id=call["id"], name=call["name"])
        result = handler(request)
        if call["name"] in STATE_CHANGING_TOOLS:
            generation[0] += 1
        if isinstance(result, ToolMessage):
            if n > 1 and last_result.get(key) == result.content:
                result = ToolMessage(
                    content=f"{result.content}\n\n[note] You already made this exact call; the result is "
                            "unchanged. Act on it instead of repeating the call.",
                    tool_call_id=result.tool_call_id, name=result.name)
            last_result[key] = result.content
        return result

    return guard


def cap_context(max_chars: int, keep_recent: int = 6):
    """Before each model call, blank the oldest tool results until the conversation fits in
    max_chars. A hard limit on what is sent, unlike token estimates."""

    @wrap_model_call
    def cap(request, handler):
        messages = list(request.messages)
        sizes = [len(str(m.content)) + len(json.dumps(getattr(m, "tool_calls", None) or [], default=str))
                 for m in messages]
        total = sum(sizes) + len(str(request.system_prompt or ""))
        tool_idx = [i for i, m in enumerate(messages) if m.type == "tool"]
        for i in tool_idx[:-keep_recent] if len(tool_idx) > keep_recent else []:
            if total <= max_chars:
                break
            m = messages[i]
            if m.name == "load_paper_spec" or str(m.content).startswith("[cleared"):
                continue
            placeholder = f"[cleared to save context: earlier {m.name} result]"
            total -= sizes[i] - len(placeholder)
            messages[i] = ToolMessage(content=placeholder, tool_call_id=m.tool_call_id, name=m.name)
        return handler(request.override(messages=messages))

    return cap


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
    if provider == "rcac":
        middleware.append(cap_context(RCAC_CONTEXT_CHARS))
    llm = build_model(provider, model, fallbacks, think)
    return create_agent(llm, tools, system_prompt=prompt, middleware=middleware)
