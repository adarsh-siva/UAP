"""Command line entry point.

  python -m repro.cli run    --spec papers/uap/spec.yaml       # deterministic: build -> reproduce -> validate
  python -m repro.cli agent  --spec papers/uap/spec.yaml       # same loop, driven by the LangChain agent
  python -m repro.cli agent  --repo https://github.com/x/y --paper https://arxiv.org/pdf/....
                                                               # agent onboards a new paper, then reproduces it
  python -m repro.cli status                                   # reproduced values and verdicts so far
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .state import ReproState
from .tools import do_docker_build, do_load_spec, do_reproduce_metric, do_validate_metric

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cmd_run(args, state: ReproState) -> int:
    info = do_load_spec(state, args.spec)
    print(f"paper: {info['paper']}  image: {info['image']}")
    if not args.skip_build:
        b = do_docker_build(state, state.spec.dockerfile)
        print(f"build: {b['status']}  ({b.get('log', b.get('message'))})")
        if b["status"] != "OK":
            print(b.get("log_tail", ""))
            return 1
    failed = False
    for name in state.spec.metrics:
        r = do_reproduce_metric(state, name, args.container)
        if r["status"] != "OK":
            print(f"{name}: {r['status']} - {r.get('message', '')}\n{r.get('log_tail', '')}")
            failed = True
            continue
        v = do_validate_metric(state, name)
        print(f"{name:30s} reproduced={v['reproduced']:<10g} recorded={v.get('recorded')!s:<10} {v['status']}")
        failed |= v["status"] == "FAIL"
    return 1 if failed else 0


def cmd_agent(args, state: ReproState) -> int:
    from .agent import build_agent, resolve_provider
    papers_dir = os.path.abspath(args.papers_dir)
    provider = resolve_provider(args.provider)
    print(f"[llm] provider={provider} model={args.model or '(default)'}")
    agent = build_agent(state, papers_dir, provider=provider, model=args.model,
                        fallbacks=not args.no_fallbacks, think=not args.no_think)
    if args.task:
        task = args.task
    elif args.spec:
        task = f"Reproduce every metric in the spec {os.path.abspath(args.spec)} and validate each one."
    else:
        task = (f"Onboard and reproduce the paper whose code is at {args.repo}."
                + (f" The paper PDF is at {args.paper}." if args.paper else "")
                + (f" Focus on these results: {args.metrics}." if args.metrics else "")
                + f" Put its Dockerfile and spec.yaml under {papers_dir}.")
    if args.instructions:
        task += " " + args.instructions
    print(f"[task] {task}\n")
    validations_before = dict(state.data["validations"])
    spec_before = _spec_text(state)
    final_text = ""
    for update in agent.stream({"messages": [{"role": "user", "content": task}]},
                               {"recursion_limit": args.max_steps}, stream_mode="updates"):
        for node, payload in update.items():
            for msg in (payload or {}).get("messages", []):
                if msg.type == "ai":
                    for call in getattr(msg, "tool_calls", []) or []:
                        print(f"[tool call] {call['name']}({json.dumps(call['args'])[:300]})")
                    text = msg.text if isinstance(msg.text, str) else msg.text()
                    if text.strip():
                        final_text = text
                        print(f"[agent] {text}\n")
                elif msg.type == "tool":
                    content = str(msg.content)
                    print(f"[tool result: {msg.name}] {content[:600]}{'...' if len(content) > 600 else ''}\n")
    import re
    new_validations = {k: v for k, v in state.data["validations"].items() if validations_before.get(k) != v}
    if re.search(r"\b(PASS|FAIL)\b|✅|❌", final_text) and not new_validations:
        print("WARNING: the agent's report marks metrics as passed/failed, but no metric was validated "
              "during this run. Those claims are not real - trust the verified results below.")
    print_verified_summary(state)
    if state.data.get("spec_path"):
        changed = _spec_text(state) != spec_before
        print(f"  spec file changed during this run: {'yes' if changed else 'no'}")
    return 0


def _spec_text(state: ReproState) -> str:
    path = state.data.get("spec_path")
    if not path or not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read()


def print_verified_summary(state: ReproState) -> None:
    """What actually happened, from saved state - independent of what the agent wrote."""
    print("=" * 70)
    print("Verified results (from workspace/state.json, not from the agent's text):")
    if not state.data.get("spec_path"):
        print("  no spec loaded")
        return
    spec = state.spec
    print(f"  paper: {spec.name}   spec: {spec.path}")
    for name, m in spec.metrics.items():
        rep = state.data["reproduced"].get(state.key(name))
        val = state.data["validations"].get(state.key(name))
        print(f"  {name:24s} recorded={m.recorded!s:<8} "
              f"reproduced={(rep or {}).get('value', '-')!s:<8} "
              f"verdict={(val or {}).get('status', 'not validated')}")


def cmd_status(args, state: ReproState) -> int:
    print(json.dumps({"reproduced": state.data["reproduced"], "validations": state.data["validations"]}, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="repro")
    p.add_argument("--workspace", default=os.path.join(HERE, "workspace"))
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="build, reproduce and validate every metric without an LLM")
    r.add_argument("--spec", required=True)
    r.add_argument("--container", default="", help="running container or image (default: spec image)")
    r.add_argument("--skip-build", action="store_true")

    a = sub.add_parser("agent", help="let the LangChain agent drive the tools")
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--spec")
    g.add_argument("--repo")
    g.add_argument("--task", help="free-form task for the agent")
    a.add_argument("--paper", help="paper PDF URL (for onboarding)")
    a.add_argument("--metrics", help="which reported results to reproduce (for onboarding)")
    a.add_argument("--instructions", default="", help="extra instructions appended to the task")
    a.add_argument("--papers-dir", default=os.path.join(HERE, "papers"))
    a.add_argument("--max-steps", type=int, default=150)
    a.add_argument("--provider", default="auto", choices=["auto", "ollama", "anthropic"],
                   help="LLM backend (auto: anthropic if ANTHROPIC_API_KEY is set, else ollama)")
    a.add_argument("--model", default="", help="model name (default: qwen3:8b for ollama, claude-opus-5-5 for anthropic)")
    a.add_argument("--no-think", action="store_true", help="ollama: turn off the model's thinking mode (faster)")
    a.add_argument("--no-fallbacks", action="store_true", help="anthropic: disable server-side refusal fallbacks")

    sub.add_parser("status", help="show reproduced values and verdicts")

    args = p.parse_args(argv)
    state = ReproState(args.workspace)
    return {"run": cmd_run, "agent": cmd_agent, "status": cmd_status}[args.cmd](args, state)


if __name__ == "__main__":
    sys.exit(main())
