"""D5 ECCC weather: composite city-hour weather table and calendar features (Section V-D).

ECCC timestamps are Local Standard Time (UTC-5 all year); collisions and WYS use local clock
time with daylight saving. Weather is converted to local clock time before any join.
"""
from __future__ import annotations

import glob
from functools import reduce

import numpy as np
import pandas as pd

from .spark_session import path

HOURLY_COLS = {
    "Date/Time (LST)": "ts_lst",
    "Temp (°C)": "temp_c",
    "Rel Hum (%)": "rel_hum",
    "Precip. Amount (mm)": "precip_mm",
    "Wind Spd (km/h)": "wind_kmh",
    "Visibility (km)": "visibility_km",
    "Weather": "weather_text",
}
DAILY_COLS = {
    "Date/Time": "date",
    "Total Rain (mm)": "daily_rain_mm",
    "Total Snow (cm)": "daily_snow_cm",
    "Total Precip (mm)": "daily_precip_mm",
    "Snow on Grnd (cm)": "snow_on_ground_cm",
    "Mean Temp (°C)": "daily_mean_temp_c",
}


def read_station_hourly(cfg, prefix: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(path(cfg, "raw", "eccc_hourly", f"{prefix}_*.csv"))))
    if not files:
        raise FileNotFoundError(f"no hourly files for {prefix}")
    df = pd.concat((pd.read_csv(f, usecols=list(HOURLY_COLS), encoding="utf-8-sig") for f in files),
                   ignore_index=True).rename(columns=HOURLY_COLS)
    df["weather_text"] = df["weather_text"].replace("NA", np.nan)
    lst = pd.to_datetime(df["ts_lst"])
    # LST is a fixed UTC-5 offset; "Etc/GMT+5" is UTC-5 in POSIX sign convention.
    local = lst.dt.tz_localize("Etc/GMT+5").dt.tz_convert("America/Toronto").dt.tz_localize(None)
    df["date"] = local.dt.normalize()
    df["hour"] = local.dt.hour
    num = ["temp_c", "rel_hum", "precip_mm", "wind_kmh", "visibility_km"]
    # In the fall-back hour two LST hours map to the same local clock hour: average them.
    agg = df.groupby(["date", "hour"]).agg({**{c: "mean" for c in num}, "weather_text": "first",
                                            "ts_lst": "size"}).rename(columns={"ts_lst": "n_obs"})
    return agg


def read_station_daily(cfg, prefix: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(path(cfg, "raw", "eccc_daily", f"{prefix}_*.csv"))))
    df = pd.concat((pd.read_csv(f, usecols=list(DAILY_COLS), encoding="utf-8-sig") for f in files),
                   ignore_index=True).rename(columns=DAILY_COLS)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def _coalesce(frames: dict, order: list[str], col: str):
    """First non-missing value across stations in `order`, plus the station that supplied it."""
    vals = pd.concat([frames[s][col].rename(s) for s in order], axis=1)
    src = vals.notna().idxmax(axis=1).where(vals.notna().any(axis=1))
    return vals.bfill(axis=1).iloc[:, 0], src


def city_hour_grid(start: str, end: str) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    return pd.MultiIndex.from_product([days, range(24)], names=["date", "hour"]).to_frame(index=False)


def build_composite_weather(cfg) -> pd.DataFrame:
    w = cfg["weather"]
    st = w["stations"]
    grid = city_hour_grid(cfg["study_period"]["start"], cfg["study_period"]["end"]).set_index(["date", "hour"])
    hourly = {k: read_station_hourly(cfg, v).reindex(grid.index) for k, v in st.items()}
    daily = {k: read_station_daily(cfg, v) for k, v in st.items()}

    out = pd.DataFrame(index=grid.index)
    for col, order_key in [("precip_mm", "precip_order"), ("visibility_km", "visibility_order"),
                           ("weather_text", "weather_text_order"), ("temp_c", "temp_order")]:
        out[col], out[f"{col}_src"] = _coalesce(hourly, w[order_key], col)
    out["rel_hum"], _ = _coalesce(hourly, ["toronto_intl_a", "toronto_city_centre", "toronto_city"], "rel_hum")
    out["wind_kmh"], _ = _coalesce(hourly, ["toronto_intl_a", "toronto_city_centre"], "wind_kmh")
    n_obs = hourly["toronto_intl_a"]["n_obs"].fillna(hourly["toronto_city"]["n_obs"])
    out["dst_transition"] = n_obs.ne(1).to_numpy()   # missing (spring) or merged (fall) clock hour

    # Short gaps (e.g. the spring-forward hour) are filled from the neighbouring hour and flagged.
    for col in ["temp_c", "rel_hum", "wind_kmh", "visibility_km"]:
        miss = out[col].isna()
        out[col] = out[col].interpolate(limit=3, limit_direction="both")
        out[f"{col}_interp"] = miss & out[col].notna()
    out["temp_c_filled"] = out["temp_c_src"].ne(w["temp_order"][0]) | out["temp_c_interp"]
    out["precip_filled"] = out["precip_mm_src"].ne(w["precip_order"][0])
    out["visibility_filled"] = out["visibility_km_src"].ne(w["visibility_order"][0]) | out["visibility_km_interp"]

    # Daily totals: Toronto City first, then City Centre, then the airport.
    d = reduce(lambda a, b: a.combine_first(b),
               [daily["toronto_city"], daily["toronto_city_centre"], daily["toronto_intl_a"]])
    out = out.reset_index().merge(d, left_on="date", right_index=True, how="left")

    # Where no hourly precipitation exists, spread the daily total over the day and flag it.
    no_hourly = out["precip_mm"].isna()
    out.loc[no_hourly, "precip_mm"] = out.loc[no_hourly, "daily_precip_mm"] / 24
    out["precip_from_daily"] = no_hourly & out["precip_mm"].notna()

    return add_weather_indicators(out, cfg)


def add_weather_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    w = cfg["weather"]
    txt = df["weather_text"].fillna("").str.lower()
    wet = df["precip_mm"].fillna(0) >= w["precip_threshold_mm"]
    text_snow = txt.str.contains("snow|ice pellets|freezing")
    text_rain = txt.str.contains("rain|drizzle|thunderstorm")
    cold = df["temp_c"] <= 0
    df["snow"] = (text_snow | (wet & cold & ~text_rain)).astype("int8")
    df["rain"] = ((text_rain | (wet & ~cold)) & ~df["snow"].astype(bool)).astype("int8")
    df["precip_any"] = ((df["rain"] + df["snow"]) > 0).astype("int8")
    df["fog"] = txt.str.contains("fog|haze|mist").astype("int8")
    lo1, lo5 = w["low_vis_km"]
    df["vis_lt1"] = (df["visibility_km"] < lo1).astype("int8")
    df["vis_1to5"] = ((df["visibility_km"] >= lo1) & (df["visibility_km"] < lo5)).astype("int8")
    labels = [f"{a}_{b}" for a, b in zip(w["temp_bins"][:-1], w["temp_bins"][1:])]
    df["temp_bin"] = pd.cut(df["temp_c"], w["temp_bins"], labels=labels, right=False).astype(str)
    df["snow_on_ground"] = (df["snow_on_ground_cm"].fillna(0) > 0).astype("int8")
    # hours since the last precipitation hour (capped at one week)
    p = df["precip_any"].to_numpy()
    since = np.empty(len(p))
    last = -10_000
    for i, v in enumerate(p):
        if v:
            last = i
        since[i] = i - last
    df["hours_since_precip"] = np.minimum(since, 168)
    df["weather_cat"] = np.select([df["snow"] == 1, df["rain"] == 1, df["vis_lt1"] == 1],
                                  ["snow", "rain", "low_visibility"], "dry_clear")
    return df


def add_calendar(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Hour, day of week, month, year, Ontario statutory holidays, school days, daylight and COVID period."""
    import holidays
    from astral import LocationInfo
    from astral.sun import sun

    d = df["date"]
    df["dow"] = d.dt.dayofweek
    df["month"] = d.dt.month
    df["year"] = d.dt.year
    df["hour_of_week"] = df["dow"] * 24 + df["hour"]
    on = holidays.Canada(subdiv="ON", years=range(d.dt.year.min(), d.dt.year.max() + 1))
    hol = set(pd.to_datetime(list(on.keys())))
    df["holiday"] = d.isin(hol).astype("int8")
    # School days (approximation of the TDSB calendar): weekdays from the Tuesday after Labour Day
    # to the end of June, excluding holidays, winter break (Dec 22 - Jan 5) and March break (week 11).
    m, day = d.dt.month, d.dt.day
    in_year = ~m.isin([7, 8]) & ~((m == 9) & (day < 4))
    winter = ((m == 12) & (day >= 22)) | ((m == 1) & (day <= 5))
    march = (m == 3) & d.dt.isocalendar().week.astype(int).between(11, 11)
    df["school_day"] = ((df["dow"] < 5) & in_year & ~winter & ~march & (df["holiday"] == 0)).astype("int8")
    city = LocationInfo("Toronto", "Canada", "America/Toronto", 43.65, -79.38)
    sun_tab = {}
    for day_ in d.drop_duplicates():
        s = sun(city.observer, date=day_.date(), tzinfo="America/Toronto")
        sun_tab[day_] = (s["sunrise"].hour + s["sunrise"].minute / 60, s["sunset"].hour + s["sunset"].minute / 60)
    rise = d.map(lambda x: sun_tab[x][0])
    set_ = d.map(lambda x: sun_tab[x][1])
    mid = df["hour"] + 0.5
    df["daylight"] = ((mid >= rise) & (mid <= set_)).astype("int8")
    c = cfg["covid_period"]
    df["covid"] = d.between(c["start"], c["end"]).astype("int8")
    return df


def coverage_table(cfg) -> pd.DataFrame:
    """Share of hours with data per station and variable (reproduces Table A3 of the proposal)."""
    rows = {}
    grid = city_hour_grid(cfg["study_period"]["start"], cfg["study_period"]["end"]).set_index(["date", "hour"])
    for k, v in cfg["weather"]["stations"].items():
        h = read_station_hourly(cfg, v).reindex(grid.index)
        rows[k] = {c: round(100 * h[c].notna().mean(), 2) for c in ["precip_mm", "visibility_km", "weather_text", "temp_c"]}
    return pd.DataFrame(rows).T
