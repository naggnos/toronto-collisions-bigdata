"""Independent verification of dataset counts reported in proposal1_v2.docx.

Read-only on data/raw; writes JSON + log to data/profile/.
Counting method: physical newline count for large CSV / ZIP members
(header lines reported separately), pandas parse for D1, csv module for KSI.
"""
import csv, io, json, os, sys, time, zipfile, glob, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "profile"
OUT.mkdir(parents=True, exist_ok=True)
res = {}
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
def save():
    (OUT / "verify_results.json").write_text(json.dumps(res, indent=2, default=str))

def count_nl(fobj, bufsize=1 << 24):
    n = 0; last = b""
    while True:
        b = fobj.read(bufsize)
        if not b: break
        n += b.count(b"\n"); last = b[-1:]
    if last and last != b"\n": n += 1   # final line without newline
    return n

# 0. inventory
inv = collections.Counter(); bytes_ = collections.Counter(); seen = set()
for p in RAW.rglob("*"):
    if p.is_file() and p.name not in (".DS_Store",):
        st = p.stat(); key = (st.st_dev, st.st_ino)
        if key in seen: continue
        seen.add(key)
        top = p.relative_to(RAW).parts[0]
        inv[top] += 1; bytes_[top] += st.st_size
res["inventory"] = {k: {"files": inv[k], "bytes": bytes_[k]} for k in sorted(inv)}
res["inventory_total"] = {"files": sum(inv.values()), "bytes": sum(bytes_.values())}
log("inventory", res["inventory_total"]); save()

# 1. D1 collisions
import pandas as pd
d1 = pd.read_csv(RAW / "collisions" / "traffic-collisions-4326.csv", low_memory=False)
r = {"rows": len(d1), "cols": d1.shape[1], "columns": list(d1.columns)}
r["year_counts"] = d1["OCC_YEAR"].value_counts(dropna=False).sort_index().to_dict()
inj = d1["INJURY_COLLISIONS"].astype(str).str.upper().eq("YES")
fat = pd.to_numeric(d1["FATALITIES"], errors="coerce").fillna(0) > 0
r["injury_yes"] = int(inj.sum())
r["fatalities_gt0"] = int(fat.sum())
r["fatal_not_injury_flag"] = int((fat & ~inj).sum())
r["injury_or_fatal"] = int((inj | fat).sum()); r["injury_or_fatal_share"] = float((inj | fat).mean())
win = d1["OCC_YEAR"].astype(str).isin([str(y) for y in range(2014, 2026)])
r["rows_2014_2025"] = int(win.sum())
r["injury_or_fatal_share_2014_2025"] = float((inj | fat)[win].mean())
r["injury_values"] = d1["INJURY_COLLISIONS"].value_counts(dropna=False).to_dict()
nsa = d1["NEIGHBOURHOOD_158"].astype(str).str.upper().eq("NSA")
zero = (pd.to_numeric(d1["LAT_WGS84"], errors="coerce").fillna(0) == 0)
r["nsa"] = int(nsa.sum()); r["nsa_share"] = float(nsa.mean()); r["zero_coords"] = int(zero.sum())
r["nsa_and_no_division"] = int((nsa & d1["DIVISION"].astype(str).str.upper().isin(["NSA", "NAN", ""])).sum())
loc = ~zero
r["located"] = int(loc.sum())
r["distinct_points"] = int(d1.loc[loc, ["LAT_WGS84", "LONG_WGS84"]].drop_duplicates().shape[0])
idcol = "_id" if "_id" in d1.columns else d1.columns[0]
dupcols = [c for c in d1.columns if c not in (idcol, "geometry", "OBJECTID", "EVENT_UNIQUE_ID")]
r["exact_dups_excl_id_located"] = int(d1.loc[loc].duplicated(subset=dupcols, keep="first").sum())
outc = ["INJURY_COLLISIONS", "FTR_COLLISIONS", "PD_COLLISIONS", "FATALITIES"]
r["all_outcomes_empty"] = int(d1[outc].isna().all(axis=1).sum())
r["occ_date_example"] = str(d1["OCC_DATE"].iloc[0])
res["D1_collisions"] = r; del d1; save(); log("D1 done", r["rows"])

# 2. KSI
with open(RAW / "ksi" / "motor-vehicle-collisions-with-ksi-data-4326.csv", newline="", encoding="utf-8-sig") as f:
    rd = csv.reader(f); hdr = next(rd); ci = hdr.index("collision_id"); ai = hdr.index("accdate")
    n = 0; ids = set(); yrs = collections.Counter(); acl = collections.Counter(); aci = hdr.index("acclass")
    for row in rd:
        n += 1; ids.add(row[ci]); yrs[row[ai][:4]] += 1; acl[row[aci]] += 1
res["D2_ksi"] = {"rows": n, "cols": len(hdr), "distinct_collisions": len(ids),
                 "year_min": min(yrs), "year_max": max(yrs), "acclass": dict(acl)}
save(); log("KSI done", n)

# 3. ECCC hourly
st = {}
for f in sorted(glob.glob(str(RAW / "eccc_hourly" / "*.csv"))):
    name = Path(f).name.rsplit("_", 2)[0]
    s = st.setdefault(name, {"files": 0, "rows": 0, "precip_nonnull": 0, "weather_nonnull": 0, "temp_nonnull": 0, "vis_nonnull": 0, "first": None, "last": None})
    with open(f, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh); s["files"] += 1
        for row in rd:
            s["rows"] += 1
            if (row.get("Precip. Amount (mm)") or "").strip(): s["precip_nonnull"] += 1
            if (row.get("Weather") or "").strip() not in ("", "NA"): s["weather_nonnull"] += 1
            if (row.get("Temp (°C)") or "").strip(): s["temp_nonnull"] += 1
            if (row.get("Visibility (km)") or "").strip(): s["vis_nonnull"] += 1
            dt_ = row.get("Date/Time (LST)")
            if s["first"] is None or dt_ < s["first"]: s["first"] = dt_
            if s["last"] is None or dt_ > s["last"]: s["last"] = dt_
    if s["files"] == 1: s["columns"] = len(rd.fieldnames)
res["D5_eccc_hourly"] = st; save(); log("ECCC done")

# 4. Bike share
bs = {}
for f in sorted(glob.glob(str(RAW / "bikeshare_trips" / "*.zip"))):
    with zipfile.ZipFile(f) as z:
        tot = 0; hdrs = 0; members = 0
        for m in z.infolist():
            if m.is_dir() or not m.filename.lower().endswith(".csv") or "__MACOSX" in m.filename: continue
            with z.open(m) as fh:
                c = count_nl(fh); tot += c; hdrs += 1; members += 1
        bs[Path(f).name] = {"members": members, "lines": tot, "records_excl_headers": tot - hdrs}
    log("bikeshare", Path(f).name, bs[Path(f).name]); res["D4_bikeshare"] = bs; save()
res["D4_bikeshare_total_records"] = sum(v["records_excl_headers"] for v in bs.values()); save()

# 5. WYS (stationary + mobile): CSV and ZIP
for key, folder in (("D3a_wys_stationary", "wys_stationary_detail"), ("D3b_wys_mobile", "wys_mobile_detail")):
    out = {}
    for f in sorted((RAW / folder).iterdir()):
        if f.suffix == ".csv":
            with open(f, "rb") as fh: c = count_nl(fh)
            out[f.name] = {"lines": c, "records_excl_headers": c - 1}
        elif f.suffix == ".zip":
            with zipfile.ZipFile(f) as z:
                tot = 0; h = 0
                for m in z.infolist():
                    if m.is_dir() or "__MACOSX" in m.filename or not m.filename.lower().endswith((".csv", ".txt")): continue
                    try:
                        with z.open(m) as fh: tot += count_nl(fh)
                    except NotImplementedError:  # e.g. Deflate64: fall back to system unzip
                        import subprocess
                        pr = subprocess.Popen(["unzip", "-p", str(f), m.filename], stdout=subprocess.PIPE)
                        tot += count_nl(pr.stdout); pr.wait()
                    h += 1
            out[f.name] = {"lines": tot, "members": h, "records_excl_headers": tot - h}
        else: continue
        log(key, f.name, out[f.name]); res[key] = out; save()
    res[key + "_total_lines"] = sum(v["lines"] for v in out.values())
    res[key + "_total_records_excl_headers"] = sum(v["records_excl_headers"] for v in out.values())
    save()
log("ALL DONE")
