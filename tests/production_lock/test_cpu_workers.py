import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/production_lock_20261020/citywide_infer'))
import cpu_workers as W


def test_auto_caps():
    assert W.auto_workers(0.4, 6.0, cores=128, avail_gb=500) == 15
    assert W.auto_workers(0.6, 6.0, cores=22, avail_gb=26) == 10
    assert W.auto_workers(0.6, 6.0, cores=4, avail_gb=26) == 2
    assert W.auto_workers(0.6, 6.0, cores=2, avail_gb=0.1) == 1
    assert W.auto_workers(0.6, 6.0, cores=64, avail_gb=4) == 3


def test_resolve():
    assert W.resolve('7', 0.4) == 7 and W.resolve(0, 0.4) == 0
    assert W.resolve('auto', 0.4, 6.0) >= 1
