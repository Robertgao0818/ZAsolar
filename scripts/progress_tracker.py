"""
Record project progress and refresh the managed section in ROADMAP.md.

Typical usage:
  python scripts/progress_tracker.py --summary "Added feature X"
  python scripts/progress_tracker.py --from-last-commit --skip-duplicates
  python scripts/progress_tracker.py --next-focus "Clean up root-level file structure"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ROADMAP_PATH = ROOT / "ROADMAP.md"
ENTRIES_PATH = ROOT / "docs" / "progress_log" / "entries.json"

ROADMAP_START = "<!-- progress:roadmap:start -->"
ROADMAP_END = "<!-- progress:roadmap:end -->"

DEFAULT_NEXT_FOCUS = [
    "Repository structure cleanup: reduce root-level script clutter and group workflows by purpose.",
    "Export reviewed keep/exclude decisions into a reusable grid manifest for later tile downloads.",
]


@dataclass(frozen=True)
class ProgressEntry:
    timestamp: str
    source: str
    summary: str

    @property
    def day(self) -> str:
        return self.timestamp[:10]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_entries() -> list[ProgressEntry]:
    if not ENTRIES_PATH.exists():
        return []
    data = json.loads(ENTRIES_PATH.read_text(encoding="utf-8"))
    return [ProgressEntry(**e) for e in data]


def save_entries(entries: list[ProgressEntry]) -> None:
    ENTRIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = [{"timestamp": e.timestamp, "source": e.source, "summary": e.summary} for e in entries]
    new_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if ENTRIES_PATH.exists() and ENTRIES_PATH.read_text(encoding="utf-8") == new_text:
        return
    ENTRIES_PATH.write_text(new_text, encoding="utf-8")


def get_last_commit_entry() -> ProgressEntry:
    output = subprocess.check_output(
        ["git", "log", "-1", "--pretty=format:%H%n%s"],
        cwd=ROOT,
        text=True,
    )
    commit_hash, summary = output.splitlines()[:2]
    return ProgressEntry(
        timestamp=utc_now_iso(),
        source=f"commit:{commit_hash[:8]}",
        summary=summary.strip(),
    )


# Commits that only touch progress-tracker artefacts are catch-up commits
# created solely to clean the dirty state after the tracker re-fired.
# Recording them re-dirties the same files and triggers another catch-up,
# which is the infinite loop we are breaking.
PROGRESS_TRACKER_ARTEFACTS = {
    "ROADMAP.md",
    "docs/progress_log/entries.json",
}


def commit_touches_only(commit_hash: str, allowed: set[str]) -> bool:
    output = subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
        cwd=ROOT,
        text=True,
    )
    files = {line.strip() for line in output.splitlines() if line.strip()}
    return bool(files) and files.issubset(allowed)


def render_recent_updates(entries: list[ProgressEntry], limit: int = 8) -> list[str]:
    # Dedup by summary, keeping the most-recent occurrence. Entries are keyed
    # internally by commit hash, so `git commit --amend` (and rebases) leave an
    # orphaned entry whose hash no longer exists but whose summary is identical
    # to the rewritten commit's. Deduping at render time keeps ROADMAP idempotent
    # across history rewrites instead of printing the same line twice.
    lines: list[str] = []
    seen: set[str] = set()
    for entry in sorted(entries, key=lambda item: item.timestamp, reverse=True):
        if entry.summary in seen:
            continue
        seen.add(entry.summary)
        lines.append(f"- {entry.day}: {entry.summary}")
        if len(lines) >= limit:
            break
    return lines


def replace_or_insert_block(
    text: str,
    start_marker: str,
    end_marker: str,
    block_title: str,
    body_lines: list[str],
) -> str:
    block = "\n".join(
        [block_title, start_marker, *body_lines, end_marker]
    ).rstrip()

    if start_marker in text and end_marker in text:
        pattern = re.compile(
            rf"{re.escape(block_title)}\n{re.escape(start_marker)}.*?{re.escape(end_marker)}",
            re.DOTALL,
        )
        return pattern.sub(block, text, count=1)

    anchor = re.search(r"^## V0:.*$", text, flags=re.MULTILINE)
    if anchor:
        return text[: anchor.start()] + block + "\n\n" + text[anchor.start() :]
    return text.rstrip() + "\n\n" + block + "\n"


def update_roadmap(entries: list[ProgressEntry], next_focus: list[str]) -> None:
    original = ROADMAP_PATH.read_text(encoding="utf-8")
    body_lines = ["### Recently Completed", *render_recent_updates(entries), "", "### Next Up"]
    body_lines.extend(f"- {item}" for item in next_focus)

    new_text = replace_or_insert_block(
        text=original,
        start_marker=ROADMAP_START,
        end_marker=ROADMAP_END,
        block_title="## Execution Track",
        body_lines=body_lines,
    )
    if new_text == original:
        return
    ROADMAP_PATH.write_text(new_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record progress and refresh ROADMAP.md")
    parser.add_argument("--summary", help="Human-written progress summary")
    parser.add_argument("--source", default="manual", help="Entry source label for manual records")
    parser.add_argument("--from-last-commit", action="store_true", help="Record the latest git commit subject")
    parser.add_argument("--skip-duplicates", action="store_true", help="Skip if the same source+summary is already recorded")
    parser.add_argument(
        "--next-focus",
        action="append",
        default=None,
        help="Replace the current next-focus list. Pass multiple times for multiple bullets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_entries()
    next_focus = args.next_focus if args.next_focus is not None else list(DEFAULT_NEXT_FOCUS)

    new_entry: ProgressEntry | None = None
    skip_reason: str | None = None
    if args.from_last_commit:
        candidate = get_last_commit_entry()
        commit_hash = candidate.source.removeprefix("commit:")
        if any(e.source == candidate.source for e in entries):
            skip_reason = f"already recorded ({candidate.source})"
        elif commit_touches_only(commit_hash, PROGRESS_TRACKER_ARTEFACTS):
            skip_reason = f"self-referential progress-tracker catch-up commit ({candidate.source})"
        else:
            new_entry = candidate
    elif args.summary:
        new_entry = ProgressEntry(timestamp=utc_now_iso(), source=args.source.strip(), summary=args.summary.strip())

    if new_entry is not None:
        duplicate = any(
            entry.source == new_entry.source and entry.summary == new_entry.summary
            for entry in entries
        )
        if not (args.skip_duplicates and duplicate):
            entries.append(new_entry)

    save_entries(entries)
    update_roadmap(entries, next_focus)

    if skip_reason is not None:
        print(f"[SKIP] {skip_reason}")
    if new_entry is not None:
        print(f"[RECORDED] {new_entry.summary}")
    print(f"[ROADMAP] {ROADMAP_PATH}")


if __name__ == "__main__":
    main()
