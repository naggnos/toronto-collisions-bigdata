"""Run the E4 tool comparison (RQ4) and record run time and peak memory for every configuration.

Each configuration runs in a fresh Python process (and, for Spark, a fresh JVM). While it runs,
this script samples the resident memory of the process and all its children every 0.1 s, so the
Spark JVM is included in the peak-memory figure.

Usage:
    python scripts/run_e4.py                      # full design from config.yaml (resumable)
    python scripts/run_e4.py --pilot              # one-month pilot (25% is the smallest real subset)
    python scripts/run_e4.py --tools duckdb spark --fractions 0.25 0.5 --cores 1 4
Results are appended to results/e4/runs.csv; finished configurations are skipped on re-run.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.spark_session import ROOT, hardware_info, load_config, path  # noqa: E402

FIELDS = ["tool", "cores", "fraction", "repeat", "n_files", "rows_in", "total_s", "spark_startup_s",
          "workload_s", "peak_mem_gb", "rows_out", "sum_volume", "sum_collisions", "mean_p85", "status"]


def peak_rss(proc: subprocess.Popen, result: dict):
    peak = 0
    try:
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
                peak = max(peak, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.1)
    except psutil.NoSuchProcess:
        pass
    result["peak"] = peak


def run_one(tool, cores, fraction, timeout):
    cmd = [sys.executable, "-m", "src.e4", "--tool", tool, "--cores", str(cores),
           "--fraction", str(fraction)]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    mem = {}
    t = threading.Thread(target=peak_rss, args=(proc, mem))
    t.start()
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    t.join()
    line = next((l for l in out.splitlines() if l.startswith("E4RESULT ")), None)
    if line is None:
        status = "timeout" if proc.returncode is None or proc.returncode < 0 else "failed"
        tail = (err or "").strip().splitlines()[-3:]
        print(f"   {status}: {' | '.join(tail)}")
        return {"status": status, "peak_mem_gb": round(mem.get("peak", 0) / 1e9, 3)}
    res = json.loads(line[len("E4RESULT "):])
    res["peak_mem_gb"] = round(mem.get("peak", 0) / 1e9, 3)
    res["status"] = "ok"
    return res


def main():
    cfg = load_config()
    e = cfg["e4"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", nargs="+", default=e["tools"])
    ap.add_argument("--fractions", nargs="+", type=float, default=e["fractions"])
    ap.add_argument("--cores", nargs="+", type=int, default=e["cores"])
    ap.add_argument("--repeats", type=int, default=e["repeats"])
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per run")
    ap.add_argument("--pilot", action="store_true", help="smallest fraction, 1 repeat, cores 1 and max")
    ap.add_argument("--out", default=str(path(cfg, "e4", "runs.csv")))
    a = ap.parse_args()
    if a.pilot:
        a.fractions, a.repeats, a.cores = [min(a.fractions)], 1, [1, max(a.cores)]
        a.out = str(path(cfg, "e4", "pilot.csv"))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "hardware.json").write_text(json.dumps(hardware_info(), indent=2))
    done = set()
    if out.exists():
        with open(out) as f:
            for r in csv.DictReader(f):
                if r["status"] == "ok":
                    done.add((r["tool"], int(r["cores"]), float(r["fraction"]), int(r["repeat"])))

    configs = []
    for rep in range(1, a.repeats + 1):          # repeats are interleaved, not run back to back
        for tool, frac, cores in itertools.product(a.tools, a.fractions, a.cores):
            if tool == "pandas" and cores != 1:
                continue
            configs.append((tool, cores, frac, rep))
    todo = [c for c in configs if c not in done]
    print(f"{len(configs)} runs in design, {len(todo)} to do -> {out}")

    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for i, (tool, cores, frac, rep) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {tool:6s} cores={cores:<2d} fraction={frac:<4} repeat {rep}", flush=True)
            res = run_one(tool, cores, frac, a.timeout)
            res.update({"tool": tool, "cores": cores, "fraction": frac, "repeat": rep})
            w.writerow(res)
            f.flush()
            if res["status"] == "ok":
                print(f"   {res['total_s']:.1f} s, peak {res['peak_mem_gb']:.2f} GB, rows_out {res['rows_out']:,}")


if __name__ == "__main__":
    main()
