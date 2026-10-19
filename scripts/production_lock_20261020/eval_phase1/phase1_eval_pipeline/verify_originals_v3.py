#!/usr/bin/env python3
"""Check original artifacts exactly, and the live host transcript's original prefix."""
import hashlib
import json
from pathlib import Path
import time


def main():
    start=time.perf_counter()
    before=json.loads(Path('originals_inventory_v3.json').read_text())
    differences=[];live=[]
    for entry in before:
        path=Path(entry['path'])
        if path.name=='codex_task1c.log':
            # Host writes this session's live transcript, independently of our task files.
            with path.open('rb') as f:prefix=f.read(entry['bytes'])
            matches=hashlib.sha256(prefix).hexdigest()==entry['sha256']
            live.append({'path':str(path),'original_prefix_bytes':entry['bytes'],'original_prefix_unchanged':matches,
                         'current_bytes_at_check':path.stat().st_size,'writer':'external live session transcript logger'})
            if not matches:differences.append(str(path))
        elif not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            differences.append(str(path))
    result={'matched':not differences,'files_checked':len(before),'static_files_exactly_unchanged':len(before)-len(live),
            'differences':differences,'live_append_only_logs':live,'authorized_task_append':'LOG.md',
            'runtime_s':time.perf_counter()-start}
    with Path('originals_integrity_confirmed_v3.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    assert not differences


if __name__=='__main__':main()
