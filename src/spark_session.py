"""Project configuration, paths and the local Spark session.

Environment overrides: BD_RAW (raw data folder), BD_SPARK_MEMORY (Spark driver memory)."""
from __future__ import annotations

import os
import platform
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config" / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("BD_RAW"):
        cfg["paths"]["raw"] = os.environ["BD_RAW"]
    if os.environ.get("BD_SPARK_MEMORY"):          # e.g. 8g on a machine with less RAM
        cfg["spark"]["driver_memory"] = os.environ["BD_SPARK_MEMORY"]
    return cfg


def path(cfg: dict, key: str, *parts: str) -> Path:
    """Absolute path for a configured folder, e.g. path(cfg, "processed", "weather_hourly.parquet")."""
    p = Path(cfg["paths"][key])
    p = p if p.is_absolute() else ROOT / p
    return p.joinpath(*parts)


def get_spark(cfg: dict | None = None, master: str | None = None,
              shuffle_partitions: int | None = None, app_name: str | None = None):
    """Create (or reuse) a local SparkSession. `master` and `shuffle_partitions` can be overridden."""
    from pyspark.sql import SparkSession

    cfg = cfg or load_config()
    s = cfg["spark"]
    return (
        SparkSession.builder
        .appName(app_name or s["app_name"])
        .master(master or s["master"])
        .config("spark.driver.memory", s["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions or s["shuffle_partitions"]))
        .config("spark.sql.session.timeZone", "America/Toronto")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def hardware_info() -> dict:
    import psutil
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "memory_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "python": platform.python_version(),
    }
