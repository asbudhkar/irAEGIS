"""Compute wall time + peak CPU/GPU memory"""
import time, resource, os, fcntl
from pathlib import Path

_start_time = None
_method = None
_cohort = None
_csv_path = None

def start(method: str, cohort: str, csv_path: str = "results/runtime_summary.csv"):
    global _start_time, _method, _cohort, _csv_path
    _start_time = time.time()
    _method = method
    _cohort = cohort
    _csv_path = csv_path

def stop():
    global _start_time
    if _start_time is None:
        return
    wall = time.time() - _start_time
    hrs, rem = divmod(int(wall), 3600)
    mins, secs = divmod(rem, 60)
    wall_hms = f"{hrs:02d}:{mins:02d}:{secs:02d}"

    cpu_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cpu_mb = cpu_kb // 1024

    try:
        import torch
        if torch.cuda.is_available():
            gpu_mb = int(torch.cuda.max_memory_allocated() / 1024**2)
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            # MPS exposes current driver allocation; no peak counter is available.
            gpu_mb = int(torch.mps.driver_allocated_memory() / 1024**2)
        else:
            gpu_mb = 0
    except ImportError:
        gpu_mb = 0

    print(f"  >> {_method}/{_cohort}: {wall_hms} ({int(wall)}s), {cpu_mb} MB CPU, {gpu_mb} MB GPU")

    csv = Path(_csv_path)
    csv.parent.mkdir(parents=True, exist_ok=True)
    header = "method,cohort,wall_hms,wall_sec,peak_cpu_mb,peak_gpu_mb\n"
    row = f"{_method},{_cohort},{wall_hms},{int(wall)},{cpu_mb},{gpu_mb}\n"

    with open(csv, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if csv.stat().st_size == 0:
            f.write(header)
        f.write(row)
        fcntl.flock(f, fcntl.LOCK_UN)

    _start_time = None
