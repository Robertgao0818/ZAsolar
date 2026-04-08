#!/usr/bin/env python3
"""Frida-based libcurl hook for Allmapsoft GEID downloader.exe.

Captures every URL libcurl is asked to fetch (plus headers / UA / referer) and
writes one JSON record per line to ``--output``. Run on Windows where the GEID
GUI is installed.

Setup (one-time, on Windows):
    pip install frida-tools

Usage:
    # 1. Launch downloader.exe by hand, fill in a small task (e.g. 1 km^2,
    #    zoom 19, a historical date), but DO NOT click Start yet.
    # 2. In another shell, attach the hook:
    python scripts/imagery/windows/hook_geid_curl.py --output curl_capture.jsonl
    # 3. Click Start in GEID. Watch the JSONL grow. Ctrl-C to stop.
    #
    # Or spawn the binary under Frida from the start (catches startup curls):
    python scripts/imagery/windows/hook_geid_curl.py \\
        --spawn "C:\\allmapsoft\\geid\\downloader.exe" \\
        --output curl_capture.jsonl
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import frida


HERE = Path(__file__).resolve().parent
DEFAULT_AGENT = HERE / "hook_geid_curl.js"
DEFAULT_EXE = r"C:\allmapsoft\geid\downloader.exe"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", type=Path, default=DEFAULT_AGENT, help="Path to Frida JS agent")
    ap.add_argument("--output", type=Path, default=Path("curl_capture.jsonl"), help="JSONL output file")
    ap.add_argument("--spawn", nargs="?", const=DEFAULT_EXE, default=None,
                    help=f"Spawn downloader.exe (default path: {DEFAULT_EXE}) instead of attaching")
    ap.add_argument("--process-name", default="downloader.exe", help="Process to attach to when not spawning")
    ap.add_argument("--duration", type=float, default=None,
                    help="Auto-detach after N seconds (use when running headless / in background)")
    ap.add_argument("--ready-file", type=Path, default=None,
                    help="Touch this path once hooks are loaded (so a parent script can wait)")
    args = ap.parse_args()

    if not args.agent.exists():
        print(f"agent file not found: {args.agent}", file=sys.stderr)
        return 2

    agent_src = args.agent.read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_fp = args.output.open("a", encoding="utf-8")

    def on_message(msg, _data):
        if msg.get("type") == "send":
            payload = msg.get("payload") or {}
            payload.setdefault("captured_at", time.time())
            line = json.dumps(payload, ensure_ascii=False)
            print(line, flush=True)
            out_fp.write(line + "\n")
            out_fp.flush()
        elif msg.get("type") == "error":
            print(f"[frida-error] {msg.get('description')}", file=sys.stderr)
            stack = msg.get("stack")
            if stack:
                print(stack, file=sys.stderr)

    pid = None
    if args.spawn:
        print(f"[hook_geid_curl] spawning {args.spawn}", file=sys.stderr)
        pid = frida.spawn(args.spawn)
        session = frida.attach(pid)
    else:
        print(f"[hook_geid_curl] attaching to {args.process_name}", file=sys.stderr)
        session = frida.attach(args.process_name)

    script = session.create_script(agent_src)
    script.on("message", on_message)
    script.load()

    if pid is not None:
        frida.resume(pid)

    if args.ready_file:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.write_text(str(int(time.time())), encoding="utf-8")

    print(f"[hook_geid_curl] writing to {args.output}", file=sys.stderr)
    if args.duration:
        print(f"[hook_geid_curl] auto-detach after {args.duration:.0f}s", file=sys.stderr)
    else:
        print("[hook_geid_curl] Ctrl-C to stop", file=sys.stderr)

    stop = {"flag": False}
    def _sigint(_signum, _frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sigint)
    try:
        signal.signal(signal.SIGTERM, _sigint)
    except Exception:
        pass

    deadline = (time.monotonic() + args.duration) if args.duration else None
    try:
        while not stop["flag"]:
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.2)
    finally:
        out_fp.close()
        try:
            session.detach()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
