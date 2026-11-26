"""D1 Traffic Collisions and D2 KSI: loading and cleaning (Section IV-B of the proposal)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .spark_session import path

COLLISIONS_FILE = "collisions/traffic-collisions-4326.csv"
KSI_FILE = "ksi/motor-vehicle-collisions-with-ksi-data-4326.csv"
ROAD_USERS = ["AUTOMOBILE", "MOTORCYCLE", "PASSENGER", "BICYCLE", "PEDESTRIAN"]


def load_raw_collisions(cfg) -> pd.DataFrame:
    return pd.read_csv(path(cfg, "raw", COLLISIONS_FILE), low_memory=False)


def clean_collisions(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per collision with parsed local date/hour, severity and data-quality flags.

    * OCC_DATE is epoch milliseconds at local midnight, so it is converted to America/Toronto.
    * severe = injury flag YES or at least one fatality (680 of 683 fatal records lack the injury flag).
    * Records without coordinates (NSA, 0/0) keep lat/lon = NaN and are excluded only from spatial joins.
    * is_duplicate marks records identical to an earlier record in every field except _id.
    """
    df = raw.copy()
    local = pd.to_datetime(df["OCC_DATE"], unit="ms", utc=True).dt.tz_convert("America/Toronto")
    out = pd.DataFrame({
        "collision_id": df["_id"].astype("int64"),
        "date": local.dt.tz_localize(None).dt.normalize(),
        "hour": df["OCC_HOUR"].astype("int16"),
        "division": df["DIVISION"],
        "fatalities": pd.to_numeric(df["FATALITIES"], errors="coerce").fillna(0).astype("int16"),
        "injury_flag": df["INJURY_COLLISIONS"].eq("YES"),
        "ftr_flag": df["FTR_COLLISIONS"].eq("YES"),
        "pd_flag": df["PD_COLLISIONS"].eq("YES"),
        "hood_158": pd.to_numeric(df["HOOD_158"], errors="coerce").astype("Int16"),
        "lon": df["LONG_WGS84"].where(df["LONG_WGS84"].abs() > 1),
        "lat": df["LAT_WGS84"].where(df["LAT_WGS84"].abs() > 1),
    })
    for c in ROAD_USERS:
        out[c.lower()] = df[c].eq("YES")
    out["year"] = out["date"].dt.year.astype("int16")
    out["month"] = out["date"].dt.month.astype("int8")
    out["dow"] = out["date"].dt.dayofweek.astype("int8")   # Monday = 0
    out["severe"] = out["injury_flag"] | (out["fatalities"] > 0)
    out["has_location"] = out["lat"].notna()
    out["injury_missing"] = df["INJURY_COLLISIONS"].isna()
    # Only located records can be recognised as duplicates: records without coordinates that share
    # date, hour and division are usually different collisions (6,117 located duplicates in D1).
    out["is_duplicate"] = df.drop(columns=["_id"]).duplicated(keep="first").to_numpy() & out["has_location"].to_numpy()
    return out


def quality_summary(raw: pd.DataFrame, clean: pd.DataFrame, years=(2014, 2025)) -> pd.DataFrame:
    """Figures quoted in the proposal, recomputed from the data."""
    in_study = clean["year"].between(*years)
    rows = {
        "records (all years)": len(clean),
        f"records {years[0]}-{years[1]}": int(in_study.sum()),
        "without coordinates": int((~clean["has_location"]).sum()),
        "without coordinates (%)": round(100 * (~clean["has_location"]).mean(), 2),
        "with fatalities": int((clean["fatalities"] > 0).sum()),
        "fatal but not injury-flagged": int(((clean["fatalities"] > 0) & ~clean["injury_flag"]).sum()),
        "missing injury flag": int(clean["injury_missing"].sum()),
        f"severe share {years[0]}-{years[1]} (%)": round(100 * clean.loc[in_study, "severe"].mean(), 2),
        "located records": int(clean["has_location"].sum()),
        "distinct located points": int(clean.loc[clean["has_location"], ["lat", "lon"]].drop_duplicates().shape[0]),
        "located duplicates (all fields except id)": int(clean["is_duplicate"].sum()),
    }
    return pd.DataFrame({"value": rows})


def load_ksi(cfg) -> pd.DataFrame:
    k = pd.read_csv(path(cfg, "raw", KSI_FILE), low_memory=False)
    k["accdate"] = pd.to_datetime(k["accdate"], errors="coerce")
    return k


def annual_counts(clean: pd.DataFrame) -> pd.DataFrame:
    g = clean.groupby("year")
    return pd.DataFrame({
        "records": g.size(),
        "injury_flagged": g["injury_flag"].sum(),
        "injury_or_fatal": g["severe"].sum(),
    })


def lonlat_to_xy(lon, lat, lat0: float = 43.7):
    """Equirectangular projection to metres around Toronto (error < 0.1% over the city)."""
    r = 6_371_000.0
    lon = np.radians(np.asarray(lon, dtype=float))
    lat = np.radians(np.asarray(lat, dtype=float))
    return r * np.cos(np.radians(lat0)) * lon, r * lat
