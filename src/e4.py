"""E4 (RQ4): the same processing workload implemented in pandas, DuckDB and PySpark.

Workload (Section VII-C): read the raw 2023-2025 WYS speed-bin records for the selected months,
aggregate them to sign-hour speed metrics (volume, mean speed, 85th-percentile speed, share of
vehicles at or above the posted limit), join the result with the city-hour weather table and with
the number of collisions matched to each sign in that hour, and write the result to Parquet.

All three implementations parse timestamps from the same string positions and use identical
formulas, so their outputs can be compared row for row (row count and column totals are checked).

All tools read the same Parquet files (partitioned by year/month).

Run one configuration:  python -m src.e4 --tool duckdb --cores 4 --fraction 0.5
The full experiment is driven by scripts/run_e4.py, which also measures peak memory.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tempfile
import time

from .spark_session import load_config, path
from .wys import month_list


# ----------------------------------------------------------------------------- inputs
def input_files(cfg, fraction: float) -> list[str]:
    root = path(cfg, "processed", "wys_raw")
    files = []
    for y, m in month_list(cfg, fraction):
        files += sorted(glob.glob(str(root / f"year={y}" / f"month={m}" / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no Parquet input for fraction {fraction}; run notebook 00 first")
    return files


def side_tables(cfg) -> dict:
    p = path(cfg, "processed")
    return {"limits": str(p / "e4_sign_limits.parquet"),
            "weather": str(p / "e4_weather.parquet"),
            "collisions": str(p / "e4_sign_collisions_hourly.parquet")}


def write_side_tables(cfg, locations, weather, matches_hourly) -> dict:
    """Small inputs shared by every tool (written once, outside the timed runs).
    date columns are ISO strings so that all tools join on the same key type."""
    t = side_tables(cfg)
    locations[["sign_id", "speed_limit"]].astype({"sign_id": "int64", "speed_limit": "int64"}).to_parquet(t["limits"], index=False)
    w = weather[["date", "hour", "rain", "snow", "temp_c", "visibility_km"]].copy()
    w["date"] = w["date"].dt.strftime("%Y-%m-%d")
    w["hour"] = w["hour"].astype("int64")
    w.to_parquet(t["weather"], index=False)
    c = matches_hourly[["sign_id", "date", "hour", "collisions"]].copy()
    c["date"] = c["date"].dt.strftime("%Y-%m-%d")
    c = c.astype({"sign_id": "int64", "hour": "int64", "collisions": "int64"})
    c.to_parquet(t["collisions"], index=False)
    return t


# ----------------------------------------------------------------------------- pandas
def run_pandas(files, tables, out_path, chunksize, timings):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    def chunks():
        for f in files:
            pf = pq.ParquetFile(f)
            for b in pf.iter_batches(batch_size=chunksize, columns=["sign_id", "datetime_bin", "speed_bin", "volume"]):
                yield b.to_pandas()

    t0 = time.perf_counter()
    partial, rows_in = [], 0
    for ch in chunks():
        rows_in += len(ch)
        dt = ch["datetime_bin"].astype(str)
        g = pd.DataFrame({"sign_id": ch["sign_id"].to_numpy(),
                          "date": dt.str.slice(0, 10).to_numpy(),
                          "hour": dt.str.slice(11, 13).astype("int64").to_numpy(),
                          "lo": ch["speed_bin"].astype(str).str.extract(r"\[(\d+),", expand=False).astype("int64").to_numpy(),
                          "vol": ch["volume"].to_numpy()})
        partial.append(g.groupby(["sign_id", "date", "hour", "lo"], sort=False)["vol"].sum())
    bins = pd.concat(partial).groupby(level=[0, 1, 2, 3]).sum().reset_index()   # combine across chunks
    bins = bins.sort_values(["sign_id", "date", "hour", "lo"], ignore_index=True)
    keys = ["sign_id", "date", "hour"]
    grp = bins.groupby(keys, sort=False)["vol"]
    bins["cum"] = grp.cumsum()
    bins["total"] = grp.transform("sum")
    bins["prev"] = bins["cum"] - bins["vol"]
    target = 0.85 * bins["total"]
    hit = (bins["prev"] < target) & (bins["cum"] >= target)
    bins["p85_here"] = np.where(hit, bins["lo"] + (target - bins["prev"]) / bins["vol"] * 5.0, np.nan)
    limits = pd.read_parquet(tables["limits"])
    bins = bins.merge(limits, on="sign_id", how="left")
    bins["vol_mid"] = bins["vol"] * (bins["lo"] + 2.5)
    bins["vol_fast"] = np.where(bins["lo"] >= bins["speed_limit"], bins["vol"], 0)
    agg = bins.groupby(keys, sort=False).agg(volume=("vol", "sum"), vol_mid=("vol_mid", "sum"),
                                             p85_speed=("p85_here", "max"), vol_fast=("vol_fast", "sum"),
                                             speed_limit=("speed_limit", "first")).reset_index()
    agg["mean_speed"] = agg["vol_mid"] / agg["volume"]
    agg["share_speeding"] = agg["vol_fast"] / agg["volume"]
    agg = agg.drop(columns=["vol_mid", "vol_fast"])
    out = (agg.merge(pd.read_parquet(tables["weather"]), on=["date", "hour"], how="inner")
              .merge(pd.read_parquet(tables["collisions"]), on=["sign_id", "date", "hour"], how="left"))
    out["collisions"] = out["collisions"].fillna(0).astype("int64")
    out.to_parquet(out_path, index=False)
    timings["workload_s"] = time.perf_counter() - t0
    return rows_in, summarize_pandas(out)


def summarize_pandas(out):
    return {"rows_out": int(len(out)), "sum_volume": int(out["volume"].sum()),
            "sum_collisions": int(out["collisions"].sum()), "mean_p85": round(float(out["p85_speed"].mean()), 4)}


# ----------------------------------------------------------------------------- DuckDB
DUCK_SQL = """
WITH raw AS (
    SELECT sign_id::BIGINT AS sign_id, substr(datetime_bin, 1, 10) AS date,
           CAST(substr(datetime_bin, 12, 2) AS BIGINT) AS hour,
           CAST(regexp_extract(speed_bin, '\\[(\\d+),', 1) AS BIGINT) AS lo, volume::BIGINT AS vol
    FROM {reader}
),
bins AS (SELECT sign_id, date, hour, lo, sum(vol) AS vol FROM raw GROUP BY ALL),
w AS (
    SELECT b.*, l.speed_limit,
           sum(vol) OVER (PARTITION BY sign_id, date, hour ORDER BY lo ROWS UNBOUNDED PRECEDING) AS cum,
           sum(vol) OVER (PARTITION BY sign_id, date, hour) AS total
    FROM bins b LEFT JOIN read_parquet('{limits}') l USING (sign_id)
),
agg AS (
    SELECT sign_id, date, hour, sum(vol) AS volume,
           sum(vol * (lo + 2.5)) / sum(vol) AS mean_speed,
           max(CASE WHEN cum - vol < 0.85 * total AND cum >= 0.85 * total
                    THEN lo + (0.85 * total - (cum - vol)) / vol * 5.0 END) AS p85_speed,
           sum(CASE WHEN lo >= speed_limit THEN vol ELSE 0 END) / sum(vol) AS share_speeding,
           first(speed_limit) AS speed_limit
    FROM w GROUP BY ALL
)
SELECT a.*, wx.rain, wx.snow, wx.temp_c, wx.visibility_km, coalesce(c.collisions, 0) AS collisions
FROM agg a
JOIN read_parquet('{weather}') wx USING (date, hour)
LEFT JOIN read_parquet('{collisions}') c USING (sign_id, date, hour)
"""


def run_duckdb(files, tables, out_path, cores, timings):
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET threads = {cores}")
    con.execute("SET preserve_insertion_order = false")
    tmp = tempfile.mkdtemp(prefix="duck_spill_")
    con.execute(f"SET temp_directory = '{tmp}'")
    flist = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    reader = f"read_parquet({flist})"
    sql = DUCK_SQL.format(reader=reader, **tables)
    t0 = time.perf_counter()
    con.execute(f"COPY ({sql}) TO '{out_path}' (FORMAT parquet)")
    timings["workload_s"] = time.perf_counter() - t0
    s = con.execute(f"""SELECT count(*), sum(volume), sum(collisions), avg(p85_speed) FROM read_parquet('{out_path}')""").fetchone()
    rows_in = con.execute(f"SELECT count(*) FROM {reader}").fetchone()[0]
    con.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return rows_in, {"rows_out": int(s[0]), "sum_volume": int(s[1]), "sum_collisions": int(s[2]),
                     "mean_p85": round(float(s[3]), 4)}


# ----------------------------------------------------------------------------- Spark
def spark_workload(spark, files, tables):
    """Returns (output DataFrame, raw DataFrame). Same steps as DUCK_SQL, written with the DataFrame API."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    raw = spark.read.parquet(*files)
    r = raw.select(F.col("sign_id").cast("long").alias("sign_id"),
                   F.substring("datetime_bin", 1, 10).alias("date"),
                   F.substring("datetime_bin", 12, 2).cast("long").alias("hour"),
                   F.regexp_extract("speed_bin", r"\[(\d+),", 1).cast("long").alias("lo"),
                   F.col("volume").cast("long").alias("vol"))
    keys = ["sign_id", "date", "hour"]
    bins = r.groupBy(*keys, "lo").agg(F.sum("vol").alias("vol"))
    limits = spark.read.parquet(tables["limits"])
    w_cum = Window.partitionBy(*keys).orderBy("lo").rowsBetween(Window.unboundedPreceding, 0)
    w_all = Window.partitionBy(*keys)
    b = (bins.join(F.broadcast(limits), "sign_id", "left")
             .withColumn("cum", F.sum("vol").over(w_cum))
             .withColumn("total", F.sum("vol").over(w_all)))
    t, prev = 0.85 * F.col("total"), F.col("cum") - F.col("vol")
    agg = b.groupBy(*keys).agg(
        F.sum("vol").alias("volume"),
        (F.sum(F.col("vol") * (F.col("lo") + 2.5)) / F.sum("vol")).alias("mean_speed"),
        F.max(F.when((prev < t) & (F.col("cum") >= t), F.col("lo") + (t - prev) / F.col("vol") * 5.0)).alias("p85_speed"),
        (F.sum(F.when(F.col("lo") >= F.col("speed_limit"), F.col("vol")).otherwise(0)) / F.sum("vol")).alias("share_speeding"),
        F.first("speed_limit", ignorenulls=True).alias("speed_limit"))
    weather = spark.read.parquet(tables["weather"])
    coll = spark.read.parquet(tables["collisions"])
    out = (agg.join(F.broadcast(weather), ["date", "hour"], "inner")
              .join(coll, keys, "left").fillna(0, subset=["collisions"]))
    return out, raw


def run_spark(files, tables, out_path, cores, cfg, timings):
    from pyspark.sql import functions as F

    from .spark_session import get_spark
    t0 = time.perf_counter()
    spark = get_spark(cfg, master=f"local[{cores}]", shuffle_partitions=max(8, 4 * cores), app_name="e4")
    spark.sparkContext.setLogLevel("ERROR")
    timings["spark_startup_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    out, raw = spark_workload(spark, files, tables)
    out.write.mode("overwrite").parquet(out_path)
    timings["workload_s"] = time.perf_counter() - t1
    res = spark.read.parquet(out_path).agg(F.count("*"), F.sum("volume"), F.sum("collisions"), F.avg("p85_speed")).collect()[0]
    rows_in = raw.count()
    spark.stop()
    return rows_in, {"rows_out": int(res[0]), "sum_volume": int(res[1]), "sum_collisions": int(res[2]),
                     "mean_p85": round(float(res[3]), 4)}


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", choices=["pandas", "duckdb", "spark"], required=True)
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--fraction", type=float, default=1.0)
    a = ap.parse_args()
    cfg = load_config()
    files = input_files(cfg, a.fraction)
    tables = side_tables(cfg)
    out_dir = tempfile.mkdtemp(prefix=f"e4_{a.tool}_")
    out_path = os.path.join(out_dir, "out.parquet")
    timings = {"spark_startup_s": 0.0}
    if a.tool == "pandas":
        rows_in, summary = run_pandas(files, tables, out_path, cfg["e4"]["pandas_chunksize"], timings)
    elif a.tool == "duckdb":
        rows_in, summary = run_duckdb(files, tables, out_path, a.cores, timings)
    else:
        rows_in, summary = run_spark(files, tables, out_path, a.cores, cfg, timings)
    # total = start-up + workload; row counting for the summary is excluded from both
    total = timings["spark_startup_s"] + timings["workload_s"]
    shutil.rmtree(out_dir, ignore_errors=True)
    print("E4RESULT " + json.dumps({"tool": a.tool, "cores": a.cores, "fraction": a.fraction,
                                    "n_files": len(files), "rows_in": int(rows_in), "total_s": total, **timings, **summary}))


if __name__ == "__main__":
    main()
