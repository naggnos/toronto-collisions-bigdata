"""D3 Watch Your Speed (stationary / school-zone signs): conversion and sign-hour speed metrics in Spark.

Raw rows are vehicle counts per sign, clock hour and 5 km/h speed bin, e.g. "[25,30)" or "[100,)".
WYS timestamps are local clock time with daylight saving; two formats occur
("2024-10-23 00:00:00.000" and "2022-01-01T00:00"), so only the first 13 characters are parsed.
"""
from __future__ import annotations

import glob
import subprocess
from pathlib import Path

import pandas as pd

from .spark_session import path

RAW_COLS = ["sign_id", "address", "dir", "datetime_bin", "speed_bin", "volume"]
LOCATIONS_FILE = "wys_locations/stationary-sign-locations-4326.csv"


def raw_schema():
    from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType
    return StructType([
        StructField("sign_id", LongType()), StructField("address", StringType()),
        StructField("dir", StringType()), StructField("datetime_bin", StringType()),
        StructField("speed_bin", StringType()), StructField("volume", IntegerType()),
    ])


def yearly_csv_files(cfg, years) -> list[str]:
    return [str(path(cfg, "raw", "wys_stationary_detail", f"stationary-{y}.csv")) for y in years]


def extract_archive_years(cfg, years) -> list[str]:
    """Optional 2017-2022 extension: unzip the yearly archives (some members use Deflate64, so the
    system `unzip` is used rather than Python's zipfile)."""
    out_dir = path(cfg, "interim", "wys_archive_csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for y in years:
        for z in glob.glob(str(path(cfg, "raw", "wys_stationary_detail", f"*stationary_detailed_{y}.zip"))):
            subprocess.run(["unzip", "-o", "-j", "-q", z, "*.csv", "-d", str(out_dir)], check=True)
        files += sorted(glob.glob(str(out_dir / f"wys_stationary_detailed_{y}*.csv")))
    return files


def add_time_columns(df):
    """Parse the clock hour and add date, hour, year and month columns."""
    from pyspark.sql import functions as F
    ts = F.to_timestamp(F.regexp_replace(F.substring("datetime_bin", 1, 13), "T", " "), "yyyy-MM-dd HH")
    return (df.withColumn("ts", ts)
              .withColumn("date", F.to_date("ts"))
              .withColumn("hour", F.hour("ts"))
              .withColumn("year", F.year("ts"))
              .withColumn("month", F.month("ts")))


def convert_to_parquet(spark, cfg, years=None, include_archive_years=()):
    """One-off conversion (not timed): raw yearly CSV -> Parquet partitioned by year/month.
    All later steps, including every E4 run, read this Parquet copy."""
    from pyspark.sql import functions as F
    years = years or cfg["speed_years"]
    files = yearly_csv_files(cfg, years) + extract_archive_years(cfg, include_archive_years)
    df = spark.read.csv(files, header=True, schema=raw_schema(), quote='"', escape='"')
    df = add_time_columns(df).filter(F.col("year").isin(list(years) + list(include_archive_years)))
    df = df.select(*RAW_COLS, "year", "month")
    pq = str(path(cfg, "processed", "wys_raw"))
    df.repartition("year", "month").write.mode("overwrite").partitionBy("year", "month").parquet(pq)
    return pq


def month_list(cfg, fraction: float) -> list[tuple[int, int]]:
    """Complete months used for a data fraction. Every k-th month is taken so that each subset
    covers all seasons (25% -> every 4th month of 2023-2025)."""
    months = [(y, m) for y in cfg["speed_years"] for m in range(1, 13)]
    n = round(len(months) * fraction)
    if n >= len(months):
        return months
    step = len(months) / n
    return [months[int(i * step)] for i in range(n)]


def load_locations(cfg) -> pd.DataFrame:
    """One row per sign: coordinates, posted limit, active period."""
    loc = pd.read_csv(path(cfg, "raw", LOCATIONS_FILE))
    xy = loc["geometry"].str.extract(r"\[\[\s*(-?[\d.]+),\s*(-?[\d.]+)\]\]").astype(float)
    loc["lon"], loc["lat"] = xy[0], xy[1]
    loc["start_date"] = pd.to_datetime(loc["start_date"]).dt.normalize()
    loc["end_date"] = pd.to_datetime(loc["end_date"]).dt.normalize()
    loc = loc.sort_values("start_date").drop_duplicates("sign_id", keep="last")
    return loc[["sign_id", "address", "dir", "lon", "lat", "speed_limit", "start_date", "end_date", "ward_no"]]


def sign_hour_metrics(raw_df, limits_df):
    """Spark: aggregate speed-bin rows to one row per sign and clock hour.

    Output: sign_id, date, hour, volume, mean_speed, p85_speed, share_speeding, speed_limit.
    * mean speed uses bin midpoints (the open top bin [100,) is treated as 100-105 km/h);
    * the 85th percentile is interpolated linearly inside the bin where the cumulative share
      first reaches 85%;
    * share_speeding = vehicles in bins whose lower bound is at or above the posted limit.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    df = add_time_columns(raw_df) if "date" not in raw_df.columns else raw_df
    lo = F.regexp_extract("speed_bin", r"\[(\d+),", 1).cast("int")
    bins = (df.withColumn("lo", lo)
              .groupBy("sign_id", "date", "hour", "lo").agg(F.sum("volume").alias("vol")))
    keys = ["sign_id", "date", "hour"]
    w_cum = Window.partitionBy(*keys).orderBy("lo").rowsBetween(Window.unboundedPreceding, 0)
    w_all = Window.partitionBy(*keys)
    b = (bins.withColumn("cum", F.sum("vol").over(w_cum))
             .withColumn("total", F.sum("vol").over(w_all))
             .withColumn("target", 0.85 * F.col("total"))
             .withColumn("prev", F.col("cum") - F.col("vol"))
             .withColumn("p85_here", F.when((F.col("prev") < F.col("target")) & (F.col("cum") >= F.col("target")),
                                             F.col("lo") + (F.col("target") - F.col("prev")) / F.col("vol") * 5.0)))
    b = b.join(F.broadcast(limits_df.select("sign_id", "speed_limit")), "sign_id", "left")
    return (b.groupBy(*keys).agg(
                F.sum("vol").alias("volume"),
                (F.sum(F.col("vol") * (F.col("lo") + 2.5)) / F.sum("vol")).alias("mean_speed"),
                F.max("p85_here").alias("p85_speed"),
                (F.sum(F.when(F.col("lo") >= F.col("speed_limit"), F.col("vol")).otherwise(0)) / F.sum("vol"))
                    .alias("share_speeding"),
                F.first("speed_limit", ignorenulls=True).alias("speed_limit")))
