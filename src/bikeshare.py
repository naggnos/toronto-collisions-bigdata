"""D4 Bike Share Toronto trips: extraction, normalisation and neighbourhood-hour trip counts (Spark).

Headers and timestamp formats change between years:
  2019-2023  "Trip Id, Trip  Duration, Start Station Id, Start Time, ..."  with "MM/dd/yyyy HH:mm"
  2024-2026  "Trip_Id, Trip_Duration, Start_Station_Id, Start_Time, ..."   with "yyyy-MM-dd HH:mm:ss"
2017-2018 files are not used: part of 2017 has no station IDs and day/month order is inconsistent.
"""
from __future__ import annotations

import glob
import json
import re
import subprocess

import pandas as pd

from .spark_session import path


def extract_trip_csvs(cfg, years) -> list[str]:
    out = path(cfg, "interim", "bikeshare_csv")
    files = []
    for y in years:
        d = out / str(y)
        if not any(d.glob("*.csv")):
            d.mkdir(parents=True, exist_ok=True)
            z = path(cfg, "raw", "bikeshare_trips", f"bikeshare-ridership-{y}.zip")
            if z.exists():
                subprocess.run(["unzip", "-o", "-j", "-q", str(z), "*.csv", "-x", "__MACOSX/*", "-d", str(d)], check=True)
            else:  # already-extracted CSV (used for small development samples)
                for f in glob.glob(str(path(cfg, "raw", "bikeshare_trips", f"bikeshare-ridership-{y}*.csv"))):
                    (d / f.split("/")[-1]).symlink_to(f)
        files += sorted(str(p) for p in d.glob("*.csv"))
    return files


def _norm(c: str) -> str:
    return re.sub(r"[^a-z]+", "_", c.lower().replace("﻿", "")).strip("_")


def load_trips(spark, cfg, years=None):
    """Spark DataFrame: trip_id, start_station_id, start_ts (local), date, hour, year."""
    from functools import reduce

    from pyspark.sql import functions as F

    years = years or cfg["bikeshare_years"]
    parts = []
    for f in extract_trip_csvs(cfg, years):
        df = spark.read.csv(f, header=True, inferSchema=False)
        df = df.toDF(*[_norm(c) for c in df.columns])
        parts.append(df.select(F.col("trip_id").cast("long"),
                               F.col("start_station_id").cast("double").cast("long").alias("start_station_id"),
                               F.col("start_time").alias("start_raw")))
    df = reduce(lambda a, b: a.unionByName(b), parts)
    ts = F.coalesce(F.to_timestamp("start_raw", "MM/dd/yyyy HH:mm"),
                    F.to_timestamp("start_raw", "yyyy-MM-dd HH:mm:ss"),
                    F.to_timestamp("start_raw", "MM/dd/yyyy HH:mm:ss"))
    return (df.withColumn("start_ts", ts).drop("start_raw")
              .withColumn("date", F.to_date("start_ts")).withColumn("hour", F.hour("start_ts"))
              .withColumn("year", F.year("start_ts"))
              .filter(F.col("year").isin(list(years))))


def stations_with_neighbourhood(cfg) -> pd.DataFrame:
    """Current Bike Share stations (GBFS snapshot) assigned to the 158 official neighbourhoods."""
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree

    f = sorted(glob.glob(str(path(cfg, "raw", "bikeshare_stations", "station_information_*.json"))))[-1]
    st = pd.DataFrame(json.load(open(f))["data"]["stations"])[["station_id", "name", "lat", "lon", "capacity"]]
    st["station_id"] = st["station_id"].astype("int64")
    nb = pd.read_csv(path(cfg, "raw", "neighbourhoods", "neighbourhoods-4326.csv"))
    polys = [shape(json.loads(g)) for g in nb["geometry"]]
    tree = STRtree(polys)
    codes = []
    for lon, lat in zip(st["lon"], st["lat"]):
        idx = tree.query(Point(lon, lat), predicate="within")
        codes.append(int(nb["AREA_SHORT_CODE"].iloc[idx[0]]) if len(idx) else None)
    st["hood_158"] = pd.array(codes, dtype="Int16")
    return st


def neighbourhood_hour_trips(trips, stations_sdf):
    """Spark: trips per neighbourhood and clock hour (start station location), plus matching stats."""
    from pyspark.sql import functions as F
    j = trips.join(F.broadcast(stations_sdf.select(F.col("station_id").alias("start_station_id"), "hood_158")),
                   "start_station_id", "left")
    counts = (j.filter(F.col("hood_158").isNotNull())
                .groupBy("hood_158", "date", "hour").agg(F.count("*").alias("trips")))
    match = j.groupBy("year").agg(F.count("*").alias("trips"),
                                  F.avg(F.col("hood_158").isNotNull().cast("double")).alias("share_matched"))
    return counts, match
