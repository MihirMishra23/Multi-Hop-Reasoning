#!/usr/bin/env python3
"""Idempotent in-place patch: add `</answer>` as a rollout stop string in
verl's ToolAgentLoop sampling_params construction.

Why this is needed:
  verl 0.7.0 builds sampling_params in `agent_loop.py::AgentLoopWorker._run_agent_loop`
  without any `stop` field. Hermes-format only emits `<|im_end|>` after `</answer>`
  to terminate a turn. When the policy collapses (loses the ability to emit
  `<|im_end|>`), the model spam-emits `<answer>X</answer><answer>X</answer>...`
  until `max_response_length` truncates. Adding `</answer>` as a stop string
  bounds the symptom and gives a cleaner reward signal.

Usage:
  python3 patch_agent_loop.py /path/to/agent_loop.py
"""

import re
import sys


def main(path: str) -> int:
    with open(path) as f:
        src = f.read()

    if 'stop=["</answer>"]' in src or "stop=['</answer>']" in src:
        print(f"[patch_agent_loop] Already patched: {path}")
        return 0

    pattern = re.compile(
        r"(\n(?P<indent>[ \t]+)sampling_params\s*=\s*dict\(\s*\n"
        r"(?:[ \t]+[a-zA-Z_][a-zA-Z0-9_]*\s*=[^\n]+\n)+"
        r")"
        r"(?P<close_indent>[ \t]+)\)",
    )
    m = pattern.search(src)
    if not m:
        print(
            f"[patch_agent_loop] FAILED: sampling_params=dict(...) pattern not "
            f"found in {path}",
            file=sys.stderr,
        )
        return 2

    inject = m.group(1) + m.group("close_indent") + '    stop=["</answer>"],\n' + m.group("close_indent") + ")"
    new_src = src[: m.start()] + inject + src[m.end() :]

    if 'stop=["</answer>"]' not in new_src:
        print(
            "[patch_agent_loop] FAILED: injection did not produce expected token",
            file=sys.stderr,
        )
        return 3

    with open(path, "w") as f:
        f.write(new_src)
    print(f"[patch_agent_loop] OK: added stop=['</answer>'] in {path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/agent_loop.py", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
