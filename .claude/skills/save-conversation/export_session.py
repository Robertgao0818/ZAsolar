#!/usr/bin/env python3
"""Convert a Claude Code session JSONL into a clean readable markdown file.

Usage:
    python3 export_session.py <output.md>
        # auto-finds the most recently modified jsonl in
        # ~/.claude/projects/-home-gaosh-projects-ZAsolar/

    python3 export_session.py <output.md> --session <session-id>
        # use a specific session

The output is a chronological transcript of user ↔ assistant turns with
tool calls + tool results folded into <details> blocks. Auto-injected
system reminders, file-history snapshots, and other harness noise are
stripped so the result reads as a real conversation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path.home() / ".claude" / "projects" / "-home-gaosh-projects-ZAsolar"

# Drop these injected blocks from user messages — they're harness noise, not
# things the user actually typed.
NOISE_PATTERNS = [
    re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL),
    re.compile(r"<task-notification>.*?</task-notification>", re.DOTALL),
    re.compile(r"<command-name>.*?</command-name>", re.DOTALL),
    re.compile(r"<command-message>.*?</command-message>", re.DOTALL),
    re.compile(r"<command-args>.*?</command-args>", re.DOTALL),
    re.compile(r"<local-command-stdout>.*?</local-command-stdout>", re.DOTALL),
    re.compile(r"<local-command-caveat>.*?</local-command-caveat>", re.DOTALL),
]

MAX_TOOL_OUTPUT_LINES = 40  # truncate tool results past this many lines


def find_latest_session() -> Path:
    if not PROJECT_DIR.exists():
        raise SystemExit(f"project conversations dir not found: {PROJECT_DIR}")
    jsonls = sorted(PROJECT_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not jsonls:
        raise SystemExit(f"no .jsonl sessions found in {PROJECT_DIR}")
    return jsonls[-1]


def clean_user_text(text: str) -> str:
    """Strip harness-injected blocks; collapse trailing whitespace."""
    for pat in NOISE_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


def block_to_markdown(block: dict) -> str:
    """Render one assistant content block as markdown."""
    btype = block.get("type")
    if btype == "text":
        return block.get("text", "").strip()
    if btype == "tool_use":
        name = block.get("name", "?")
        inp = block.get("input", {})
        # Pretty-print short inputs inline; long inputs in <details>
        try:
            inp_json = json.dumps(inp, indent=2, ensure_ascii=False)
        except Exception:
            inp_json = repr(inp)
        if len(inp_json) < 200:
            return f"_🔧 {name}_ `{inp_json}`"
        return (
            f"<details><summary>🔧 <b>{name}</b></summary>\n\n"
            f"```json\n{inp_json}\n```\n\n</details>"
        )
    if btype == "thinking":
        # Skip internal thinking blocks — they're not part of the conversation
        return ""
    return ""


def tool_result_to_markdown(content) -> str:
    """Render a tool_result block (as found inside user messages)."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        text = "\n".join(parts)
    else:
        text = str(content)

    text = text.rstrip()
    if not text:
        return ""

    lines = text.splitlines()
    if len(lines) > MAX_TOOL_OUTPUT_LINES:
        head = "\n".join(lines[: MAX_TOOL_OUTPUT_LINES // 2])
        tail = "\n".join(lines[-MAX_TOOL_OUTPUT_LINES // 2:])
        text = f"{head}\n\n... [{len(lines) - MAX_TOOL_OUTPUT_LINES} lines elided] ...\n\n{tail}"

    return (
        "<details><summary>📤 tool result</summary>\n\n"
        f"```\n{text}\n```\n\n</details>"
    )


def render_session(jsonl_path: Path, tldr: str = "") -> str:
    out: list[str] = []
    out.append(f"# Claude Code session — {jsonl_path.stem}\n")
    out.append(f"_Exported from `{jsonl_path}`_\n")
    if tldr.strip():
        out.append(f"## TL;DR\n\n{tldr.strip()}\n")
        out.append("---\n")

    first_ts = None
    last_ts = None
    user_count = 0
    assistant_count = 0

    pending_assistant_text: list[str] = []

    def flush_assistant():
        nonlocal pending_assistant_text
        if pending_assistant_text:
            text = "\n\n".join(t for t in pending_assistant_text if t)
            if text.strip():
                out.append(f"### 🤖 Assistant\n\n{text}\n")
            pending_assistant_text = []

    with jsonl_path.open() as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = ev.get("timestamp")
            if ts:
                first_ts = first_ts or ts
                last_ts = ts

            etype = ev.get("type")
            if etype not in ("user", "assistant"):
                continue
            msg = ev.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")

            if etype == "user":
                # Content can be a string or a list of blocks (tool_results etc.)
                if isinstance(content, str):
                    text = clean_user_text(content)
                    if text:
                        flush_assistant()
                        out.append(f"### 🧑 User\n\n{text}\n")
                        user_count += 1
                elif isinstance(content, list):
                    user_text_parts = []
                    tool_result_parts = []
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text":
                            cleaned = clean_user_text(block.get("text", ""))
                            if cleaned:
                                user_text_parts.append(cleaned)
                        elif btype == "tool_result":
                            tr = tool_result_to_markdown(block.get("content", ""))
                            if tr:
                                tool_result_parts.append(tr)
                    if user_text_parts:
                        flush_assistant()
                        out.append(f"### 🧑 User\n\n" + "\n\n".join(user_text_parts) + "\n")
                        user_count += 1
                    if tool_result_parts:
                        # Tool results are responses to assistant tool calls — keep with assistant turn
                        pending_assistant_text.extend(tool_result_parts)

            elif etype == "assistant":
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            md = block_to_markdown(block)
                            if md:
                                pending_assistant_text.append(md)
                                if block.get("type") == "text":
                                    assistant_count += 1
                elif isinstance(content, str):
                    pending_assistant_text.append(content.strip())
                    assistant_count += 1

    flush_assistant()

    # Insert a header summary line near the top
    summary = (
        f"_{user_count} user turns · {assistant_count} assistant turns"
        + (f" · {first_ts} → {last_ts}" if first_ts else "")
        + "_\n"
    )
    out.insert(2, summary)

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", type=Path, help="output markdown path")
    ap.add_argument("--session", type=str, default=None,
                    help="session ID (default: most recently modified jsonl)")
    ap.add_argument("--tldr", type=str, default="",
                    help="optional TL;DR markdown to inject under the title")
    ap.add_argument("--tldr-file", type=Path, default=None,
                    help="read TL;DR from a file (overrides --tldr)")
    args = ap.parse_args()

    if args.session:
        src = PROJECT_DIR / f"{args.session}.jsonl"
        if not src.exists():
            raise SystemExit(f"session not found: {src}")
    else:
        src = find_latest_session()

    tldr = args.tldr
    if args.tldr_file:
        tldr = args.tldr_file.read_text()

    print(f"source : {src}", file=sys.stderr)
    md = render_session(src, tldr=tldr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    print(f"wrote  : {args.output} ({len(md):,} chars)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
