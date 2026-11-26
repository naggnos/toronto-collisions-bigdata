"""Grid-based spatial join (Section V-D): match each collision to the nearest active WYS sign
within a buffer. Points are projected to metres and binned into square cells whose side equals
the buffer, so each collision is compared only with signs in its own and the 8 neighbouring cells
instead of with every sign."""
from __future__ import annotations

import pandas as pd

from .collisions import lonlat_to_xy


def add_xy_cell(df: pd.DataFrame, cell_m: float) -> pd.DataFrame:
    df = df.copy()
    df["x"], df["y"] = lonlat_to_xy(df["lon"], df["lat"])
    df["cx"] = (df["x"] // cell_m).astype("int64")
    df["cy"] = (df["y"] // cell_m).astype("int64")
    return df


def match_collisions_to_signs(spark, collisions: pd.DataFrame, signs: pd.DataFrame, buffer_m: float):
    """Spark DataFrame: collision_id, sign_id, dist_m for the nearest sign within buffer_m that was
    active on the collision date (start_date <= date and (end_date is null or date <= end_date))."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    c = add_xy_cell(collisions.loc[collisions["has_location"],
                                   ["collision_id", "date", "lon", "lat"]], buffer_m)
    s = add_xy_cell(signs[["sign_id", "lon", "lat", "start_date", "end_date"]].dropna(subset=["lon"]), buffer_m)
    cs = spark.createDataFrame(c.drop(columns=["lon", "lat"]))
    ss = spark.createDataFrame(s.drop(columns=["lon", "lat"])
                                .rename(columns={"x": "sx", "y": "sy", "cx": "scx", "cy": "scy"}))
    offsets = spark.createDataFrame([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], ["dx", "dy"])
    cand = (cs.crossJoin(F.broadcast(offsets))
              .withColumn("scx", F.col("cx") + F.col("dx")).withColumn("scy", F.col("cy") + F.col("dy"))
              .join(F.broadcast(ss), ["scx", "scy"])
              .withColumn("dist_m", F.sqrt((F.col("x") - F.col("sx")) ** 2 + (F.col("y") - F.col("sy")) ** 2))
              .filter(F.col("dist_m") <= buffer_m)
              .filter((F.col("date") >= F.col("start_date")) &
                      (F.col("end_date").isNull() | (F.col("date") <= F.col("end_date")))))
    w = Window.partitionBy("collision_id").orderBy("dist_m", "sign_id")
    return (cand.withColumn("rk", F.row_number().over(w)).filter("rk = 1")
                .select("collision_id", "sign_id", "dist_m"))


def nearest_distance_summary(collisions: pd.DataFrame, signs: pd.DataFrame) -> pd.Series:
    """Distance from each sign to its nearest collision point (feasibility check in the proposal)."""
    from scipy.spatial import cKDTree
    pts = collisions.loc[collisions["has_location"], ["lon", "lat"]].drop_duplicates()
    px, py = lonlat_to_xy(pts["lon"], pts["lat"])
    sx, sy = lonlat_to_xy(signs["lon"], signs["lat"])
    d, _ = cKDTree(list(zip(px, py))).query(list(zip(sx, sy)))
    return pd.Series(d).describe()
