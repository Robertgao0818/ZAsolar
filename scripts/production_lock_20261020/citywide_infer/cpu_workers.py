"""Auto CPU worker count: min(cores-2, floor(mem_cap_gb / per_worker_gb), available-RAM cap), >= 1.

mem_cap_gb defaults to 6 (rog limit); override with --mem-cap-gb or an explicit integer worker count.
"""
import os


def _avail_gb():
    try:
        with open('/proc/meminfo') as f:
            for ln in f:
                if ln.startswith('MemAvailable:'):
                    return int(ln.split()[1]) / 1048576
    except OSError:
        pass
    return None


def _cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def auto_workers(per_worker_gb, mem_cap_gb=6.0, cores=None, avail_gb=None):
    cores = _cores() if cores is None else cores
    avail_gb = _avail_gb() if avail_gb is None else avail_gb
    n = min(cores - 2, int(mem_cap_gb / per_worker_gb + 1e-9))
    if avail_gb is not None:
        n = min(n, int(0.5 * avail_gb / per_worker_gb + 1e-9))
    return max(1, n)


def resolve(value, per_worker_gb, mem_cap_gb=6.0):
    """value: 'auto' or an int-like; explicit ints are honoured as given."""
    if value is None or str(value).lower() == 'auto':
        return auto_workers(per_worker_gb, mem_cap_gb)
    n = int(value)
    if n < 0:
        raise ValueError('worker count must be >= 0')
    return n
