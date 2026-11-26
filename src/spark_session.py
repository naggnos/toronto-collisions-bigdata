"""Shared helpers: load the project config and start a local Spark session."""
from __future__ import annotations

import platform
from pathlib import Path

import yaml
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config" / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_spark(cfg: dict | None = None, master: str | None = None,
              shuffle_partitions: int | None = None) -> SparkSession:
    """Create (or reuse) a local SparkSession using values from config.yaml.

    `master` and `shuffle_partitions` can be overridden for performance runs.
    """
    cfg = cfg or load_config()
    s = cfg["spark"]
    return (
        SparkSession.builder
        .appName(s["app_name"])
        .master(master or s["master"])
        .config("spark.driver.memory", s["driver_memory"])
        .config("spark.sql.shuffle.partitions",
                str(shuffle_partitions or s["shuffle_partitions"]))
        .config("spark.sql.session.timeZone", "America/Toronto")
        .getOrCreate()
    )


def hardware_info() -> dict:
    """Machine details to log alongside performance results."""
    import os
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }
