#!/usr/bin/env python3
"""Format runpodctl pod JSON for display. Used by runpod_pod.sh."""
import json, sys

def fmt_status(data):
    d = data if isinstance(data, dict) else json.loads(data)
    print(f"  Status:  {d.get('desiredStatus', '?')}")
    print(f"  GPU:     {d.get('gpuCount', '?')}x (${d.get('costPerHr', '?')}/hr)")
    print(f"  Image:   {d.get('imageName', '?')}")
    print(f"  Uptime:  {d.get('uptimeSeconds', 0) // 60} min")
    ssh = d.get("ssh", {})
    if ssh.get("host"):
        print(f"  SSH:     ssh {ssh['host']} -p {ssh['port']}")
    elif ssh.get("error"):
        print(f"  SSH:     {ssh['error']}")

def fmt_ssh(data):
    """Extract SSH host and port. Prints 'host port' or empty."""
    d = data if isinstance(data, dict) else json.loads(data)
    ssh = d.get("ssh", {})
    if ssh.get("host") and ssh.get("port"):
        print(f"{ssh['host']} {ssh['port']}")
    elif ssh.get("command"):
        import re
        cmd = ssh["command"]
        m_host = re.search(r"ssh\s+(\S+@\S+)", cmd)
        m_port = re.search(r"-p\s+(\d+)", cmd)
        if m_host and m_port:
            print(f"{m_host.group(1)} {m_port.group(1)}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    data = json.load(sys.stdin)
    if cmd == "status":
        fmt_status(data)
    elif cmd == "ssh":
        fmt_ssh(data)
