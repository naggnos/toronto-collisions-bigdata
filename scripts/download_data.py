"""Download the raw datasets for the Toronto road-safety Big Data project.

Sources: City of Toronto Open Data (CKAN API), Bike Share Toronto GBFS,
Environment and Climate Change Canada (ECCC) bulk climate data.

Usage:
    python3 scripts/download_data.py --dry-run   # list planned files
    python3 scripts/download_data.py             # download (resumable)

Every file is recorded in data/raw/manifest.csv (source URL, bytes, SHA-256,
download time) so the acquisition step is reproducible.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CKAN = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/package_show?id="
LOCAL_COLLISIONS = ROOT / "traffic-collisions-4326.csv"

YEARS = range(2014, 2026)  # analysis window 2014-2025
ECCC_STATIONS = {  # station ID -> name (verified via bulk API on 2026-10-16)
    51459: "toronto_intl_a",        # Pearson, Climate ID 6158731
    31688: "toronto_city",          # Climate ID 6158355, hourly precip available
    48549: "toronto_city_centre",   # Climate ID 6158359
}

# (CKAN package, target folder, resource filter, priority: lower runs first)
TORONTO_PACKAGES = [
    ("motor-vehicle-collisions-involving-killed-or-seriously-injured-persons", "ksi",
     lambda n, f: n.endswith("4326.csv"), 1),
    ("school-safety-zone-watch-your-speed-program-locations", "wys_locations",
     lambda n, f: n.endswith("4326.csv") or "readme" in n.lower(), 1),
    ("mobile-watch-your-speed-program-speed-summary", "wys_mobile_summary",
     lambda n, f: n.lower().endswith(".csv"), 1),
    ("neighbourhoods", "neighbourhoods",
     lambda n, f: n.endswith("4326.geojson") or n.endswith("4326.csv"), 1),
    ("toronto-centreline-tcl", "centreline",
     lambda n, f: n.endswith("4326.csv"), 1),
    ("intersection-file-city-of-toronto", "intersections",
     lambda n, f: n.endswith("4326.csv"), 1),
    ("bike-share-toronto-ridership-data", "bikeshare_trips",
     lambda n, f: f in ("ZIP", "XLSX"), 2),
    ("mobile-watch-your-speed-program-detailed-speed-counts", "wys_mobile_detail",
     lambda n, f: f in ("ZIP", "CSV", "XLSX"), 3),
    ("school-safety-zone-watch-your-speed-program-detailed-speed-counts", "wys_stationary_detail",
     lambda n, f: f in ("ZIP", "CSV", "XLSX"), 4),
]

lock = threading.Lock()


def ckan_resources(pkg):
    with urllib.request.urlopen(CKAN + pkg, timeout=60) as r:
        return json.load(r)["result"]["resources"]


def filename_from_url(url, fallback):
    name = url.split("?")[0].rstrip("/").split("/")[-1]
    return name if "." in name else fallback


def build_plan():
    plan = []  # dicts: url, path, expected_size, priority
    for pkg, folder, keep, prio in TORONTO_PACKAGES:
        for res in ckan_resources(pkg):
            name, fmt, url = res["name"], (res.get("format") or "").upper(), res.get("url")
            if not url or not keep(name, fmt) or "/datastore/dump/" in url:
                continue
            fname = filename_from_url(url, name.replace(" ", "_") + "." + fmt.lower())
            size = int(res["size"]) if res.get("size") else None
            plan.append(dict(url=url, path=RAW / folder / fname, size=size, prio=prio))
    for ep in ("station_information", "system_information"):
        plan.append(dict(url=f"https://tor.publicbikesystem.net/ube/gbfs/v1/en/{ep}",
                         path=RAW / "bikeshare_stations" / f"{ep}_{dt.date.today()}.json",
                         size=None, prio=1))
    base = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&Day=1"
    for sid, sname in ECCC_STATIONS.items():
        for y in YEARS:
            plan.append(dict(url=f"{base}&stationID={sid}&Year={y}&Month=1&timeframe=2",
                             path=RAW / "eccc_daily" / f"{sname}_{sid}_{y}.csv", size=None, prio=0))
            for m in range(1, 13):
                plan.append(dict(url=f"{base}&stationID={sid}&Year={y}&Month={m}&timeframe=1",
                                 path=RAW / "eccc_hourly" / f"{sname}_{sid}_{y}_{m:02d}.csv",
                                 size=None, prio=0))
    plan.sort(key=lambda p: (p["prio"], p["size"] or 0))
    return plan


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(manifest, item, status):
    p = item["path"]
    row = dict(file=str(p.relative_to(RAW)), url=item["url"], bytes=p.stat().st_size,
               sha256=sha256(p), status=status,
               downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    with lock:
        new = not manifest.exists()
        with open(manifest, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


def fetch(item, manifest):
    p = item["path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and (item["size"] is None or p.stat().st_size == item["size"]):
        return f"skip  {p.relative_to(RAW)}"
    part = p.with_suffix(p.suffix + ".part")
    cmd = ["curl", "-sSL", "--fail", "--retry", "5", "--retry-delay", "10",
           "--connect-timeout", "30", "-C", "-", "-o", str(part), item["url"]]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode not in (0, 33):  # 33: server ignores range on a fresh file
        return f"FAIL  {p.relative_to(RAW)}  curl={res.returncode} {res.stderr.strip()[:200]}"
    part.rename(p)
    if item["size"] and p.stat().st_size != item["size"]:
        record(manifest, item, "size_mismatch")
        return f"WARN  {p.relative_to(RAW)} size {p.stat().st_size} != {item['size']}"
    record(manifest, item, "ok")
    return f"ok    {p.relative_to(RAW)}  {p.stat().st_size / 1e6:.1f} MB"


def link_local_collisions(manifest):
    dst = RAW / "collisions" / LOCAL_COLLISIONS.name
    if LOCAL_COLLISIONS.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(LOCAL_COLLISIONS, dst)  # hard link: no extra disk space
        record(manifest, dict(path=dst, url="https://open.toronto.ca/dataset/"
                              "police-annual-statistical-report-traffic-collisions/"), "local_link")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    plan = build_plan()
    known = sum(p["size"] or 0 for p in plan)
    print(f"{len(plan)} files planned; {known / 1e9:.2f} GB with known sizes "
          f"(+ ECCC/GBFS files of unknown size, small)")
    if args.dry_run:
        for p in plan:
            sz = f"{p['size'] / 1e6:9.1f} MB" if p["size"] else "        ? MB"
            print(f"  p{p['prio']} {sz}  {p['path'].relative_to(RAW)}")
        return

    manifest = RAW / "manifest.csv"
    RAW.mkdir(parents=True, exist_ok=True)
    link_local_collisions(manifest)
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(fetch, p, manifest) for p in plan]
        for i, fut in enumerate(as_completed(futures), 1):
            msg = fut.result()
            failures += msg.startswith("FAIL")
            print(f"[{i}/{len(plan)}] {msg}", flush=True)
    print(f"done; failures: {failures}. Re-run the script to resume failed/partial files.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
