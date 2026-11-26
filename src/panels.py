"""Analysis tables built with Spark: city-hour panel (RQ1), sign-hour and sign-level tables (RQ2)
and the neighbourhood-hour panel (RQ3)."""
from __future__ import annotations

import pandas as pd

WEATHER_KEEP = ["date", "hour", "temp_c", "precip_mm", "visibility_km", "rain", "snow", "precip_any",
                "fog", "vis_lt1", "vis_1to5", "temp_bin", "snow_on_ground", "hours_since_precip",
                "daily_rain_mm", "daily_snow_cm", "weather_cat", "dst_transition",
                "dow", "month", "year", "hour_of_week", "holiday", "school_day", "daylight", "covid"]


def to_spark(spark, pdf: pd.DataFrame):
    pdf = pdf.copy()
    for c in pdf.columns:
        if str(pdf[c].dtype) in ("Int16", "Int32", "Int64"):
            pdf[c] = pdf[c].astype("float64")
        elif pdf[c].dtype == bool:
            pdf[c] = pdf[c].astype("int8")
        elif str(pdf[c].dtype) in ("str", "string", "object") :
            pdf[c] = pdf[c].astype(object)
    return spark.createDataFrame(pdf)


def city_hour_panel(spark, collisions: pd.DataFrame, weather: pd.DataFrame, drop_duplicates=False):
    """Spark: collisions per city-hour (grouping) left-joined to the full weather/calendar grid (join)."""
    from pyspark.sql import functions as F
    c = collisions if not drop_duplicates else collisions[~collisions["is_duplicate"]]
    cs = to_spark(spark, c[["date", "hour", "severe", "bicycle", "pedestrian"]])
    counts = cs.groupBy("date", "hour").agg(
        F.count("*").alias("collisions"),
        F.sum("severe").alias("severe_collisions"),
        F.sum("bicycle").alias("cyclist_collisions"),
        F.sum("pedestrian").alias("pedestrian_collisions"))
    w = to_spark(spark, weather[WEATHER_KEEP])
    panel = w.join(counts, ["date", "hour"], "left").fillna(0, subset=["collisions", "severe_collisions",
                                                                       "cyclist_collisions", "pedestrian_collisions"])
    return panel.orderBy("date", "hour")


def collision_level_table(collisions: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Each collision with the weather and calendar of its city-hour (severity models)."""
    cols = ["collision_id", "date", "hour", "severe", "automobile", "motorcycle", "passenger", "bicycle",
            "pedestrian", "is_duplicate", "has_location"]
    return collisions[cols].merge(weather[WEATHER_KEEP], on=["date", "hour"], how="inner")


def sign_hour_panel(sign_hour, weather_sdf, locations_sdf, min_volume: int):
    """Spark: sign-hour speed metrics joined to weather, restricted to each sign's active period."""
    from pyspark.sql import functions as F
    s = sign_hour.filter(F.col("volume") >= min_volume).join(weather_sdf, ["date", "hour"], "inner")
    s = s.join(F.broadcast(locations_sdf.select("sign_id", "start_date", "end_date")), "sign_id", "left")
    return s.filter((F.col("date") >= F.col("start_date")) &
                    (F.col("end_date").isNull() | (F.col("date") <= F.col("end_date")))).drop("start_date", "end_date")


def sign_level_table(sign_hour, matches, collisions_sdf, years):
    """Spark: one row per sign with volume-weighted mean speeds, active days and the number of matched
    collisions on days when the sign recorded data."""
    from pyspark.sql import functions as F
    sh = sign_hour.filter(F.year("date").isin(list(years)))
    speeds = sh.groupBy("sign_id").agg(
        (F.sum(F.col("p85_speed") * F.col("volume")) / F.sum("volume")).alias("p85_speed"),
        (F.sum(F.col("share_speeding") * F.col("volume")) / F.sum("volume")).alias("share_speeding"),
        (F.sum(F.col("mean_speed") * F.col("volume")) / F.sum("volume")).alias("mean_speed"),
        F.sum("volume").alias("vehicles"),
        F.countDistinct("date").alias("active_days"),
        F.first("speed_limit", ignorenulls=True).alias("speed_limit"))
    days = sh.select("sign_id", "date").distinct()
    m = (matches.join(collisions_sdf.select("collision_id", "date", "severe"), "collision_id")
                .join(days, ["sign_id", "date"], "inner")
                .groupBy("sign_id").agg(F.count("*").alias("collisions"), F.sum("severe").alias("severe_collisions")))
    return speeds.join(m, "sign_id", "left").fillna(0, subset=["collisions", "severe_collisions"])


def neighbourhood_hour_panel(trip_counts, collisions_sdf, weather_sdf):
    """Spark: neighbourhood-hours with at least one Bike Share trip, cyclist collisions in the same
    neighbourhood and hour, and city-hour weather."""
    from pyspark.sql import functions as F
    cyc = (collisions_sdf.filter((F.col("bicycle") == 1) & F.col("hood_158").isNotNull())
                         .groupBy("hood_158", "date", "hour").agg(F.count("*").alias("cyclist_collisions")))
    return (trip_counts.join(cyc, ["hood_158", "date", "hour"], "left")
                       .fillna(0, subset=["cyclist_collisions"])
                       .join(weather_sdf, ["date", "hour"], "inner"))
