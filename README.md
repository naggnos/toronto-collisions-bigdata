# Weather, Operating Speed and Exposure: Toronto Motor-Vehicle Collisions, 2014–2025

CSCI 6886 Big Data Analytics (Dr. Salma Noor), Fairleigh Dickinson University Vancouver
Team: **Gan Song** (2095706) and **Mingye Lang** (2099150)

This repository implements the project proposed in Deliverable 1. We integrate Toronto Police collision records, City of Toronto Watch Your Speed (WYS) sign counts, Bike Share Toronto trips and Environment and Climate Change Canada (ECCC) hourly weather. We ask how weather is associated with collision frequency and severity, and to what extent measured vehicle speeds and cycling exposure explain this association. We also compare pandas, DuckDB and Apache Spark on the project's main processing workload.

## Research questions and experiments

**Primary question:** In Toronto from 2014 to 2025, how is weather associated with the frequency and severity of police-reported collisions, and to what extent do measured vehicle speeds and travel exposure explain this association?

| RQ | Question | Experiment | Model / configurations | Evaluation |
|---|---|---|---|---|
| RQ1 | How are rain, snow, low visibility and low temperature associated with collisions per hour and with the probability that a collision is severe (injury or fatal)? | E1 | Negative binomial (city-hour, checked against Poisson); logistic regression and gradient-boosted trees (collision) | IRR and odds ratios with 95% CI; MAE/RMSE against baselines; PR-AUC (primary), recall, F1, ROC-AUC, calibration |
| RQ2 | Do WYS-measured speeds (85th percentile, share speeding) decrease in adverse weather, and are higher-speed sites associated with more nearby collisions? | E2 | Linear regression with sign fixed effects, SEs clustered by sign; negative binomial per sign with active-days exposure; 150 m and 250 m buffers | Change in km/h with 95% CI; IRR per km/h; sensitivity to buffer size |
| RQ3 | Using Bike Share trips as exposure, how does the cyclist collision rate per trip vary with weather? | E3 | Negative binomial (neighbourhood-hour) with log(trips) offset, compared with a model without exposure | Cyclist collisions per 100,000 trips with 95% CI |
| RQ4 | Which of pandas, DuckDB and PySpark is most suitable for this workload? | E4 | 3 tools × 25/50/75/100% of data × 1/2/4/8/16 cores × CSV/Parquet, 3 runs each (median) | Run time, Spark start-up, peak memory, rows/s, speed-up |

Robustness checks for E1–E3: without the 6,117 duplicate records, alternative buffer radii (100/250/500 m), and excluding the COVID-19 years (2020–2021).

## Repository layout

```
├── config/config.yaml        # all paths and parameters (periods, split, buffers, E4 design, seeds)
├── src/
│   ├── collisions.py         # D1/D2 loading and cleaning
│   ├── weather.py            # composite city-hour weather table, calendar features
│   ├── wys.py                # WYS conversion and sign-hour speed metrics (Spark)
│   ├── spatial.py            # grid-based spatial join of collisions to WYS signs (Spark)
│   ├── bikeshare.py          # Bike Share trips and neighbourhood-hour counts (Spark)
│   ├── panels.py             # city-hour, sign-hour, sign-level and neighbourhood-hour tables (Spark)
│   ├── e4.py                 # E4 workload implemented in pandas, DuckDB and Spark
│   ├── models.py             # count, severity (incl. PyTorch), fixed-effects models
│   └── spark_session.py      # config loader and local Spark session
├── scripts/
│   ├── download_data.py      # downloads every raw file and writes the manifest
│   ├── verify_profile.py     # independent check of the dataset figures in the proposal
│   └── run_e4.py             # runs the E4 design and measures time and peak memory
├── notebooks/                # 00-07, run in order
├── data/                     # raw, interim and processed data (not committed)
├── results/                  # figures/, tables/, e4/
└── docs/                     # CONTRIBUTIONS.md, AI_USAGE.md
```

## Datasets

All datasets are open data. Raw files (17.3 GB) are not stored in the repository. `scripts/download_data.py` downloads them into `data/raw/` and records URL, size and SHA-256 of every file in `data/raw/manifest.csv`. See [`data/README.md`](data/README.md).

| ID | Dataset | Publisher | Used for |
|---|---|---|---|
| D1 | Traffic Collisions | Toronto Police Service | RQ1–RQ3 |
| D2 | Killed or Seriously Injured (KSI) collisions | Toronto Police Service | descriptive checks |
| D3 | Watch Your Speed stationary (school-zone) sign speed counts and locations | City of Toronto | RQ2, RQ4 |
| D4 | Bike Share Toronto ridership and station information | Toronto Parking Authority | RQ3 |
| D5 | Hourly and daily weather, three Toronto stations | ECCC | RQ1–RQ3 |

## How to run

### 1. Requirements

- Python 3.11–3.13, Java 17 or 21 (for Spark), `unzip` (some archives use Deflate64)
- About 60 GB free disk space
- The results in the report were produced on an Apple M5 Pro (18 cores, 48 GB RAM)

### 2. Set up

```bash
git clone https://github.com/naggnos/toronto-collisions-bigdata.git
cd toronto-collisions-bigdata
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Data

Either download everything:

```bash
python scripts/download_data.py
```

or point the project at an existing copy of the raw data:

```bash
ln -s "/path/to/existing/data/raw" data/raw      # or: export BD_RAW=/path/to/existing/data/raw
```

### 4. Run

| Step | What it does | Approx. time* |
|---|---|---|
| `notebooks/00_manifest_and_parquet.ipynb` | manifest check; clean collisions; composite weather; WYS → Parquet + monthly CSV | 20 min |
| `notebooks/01_data_quality_eda.ipynb` | data-quality issues and EDA figures | 3 min |
| `notebooks/02_e4_tool_comparison.ipynb` (first cells) | shared inputs for E4 | 2 min |
| `python scripts/run_e4.py --pilot` then `python scripts/run_e4.py` | **E4**: 264 timed runs (resumable) | several hours |
| `notebooks/02_e4_tool_comparison.ipynb` (rest) | E4 tables, figures and tool selection | 1 min |
| `notebooks/03_pipeline.ipynb` | Spark pipeline: sign-hour metrics, panels, spatial joins, Bike Share | 30 min |
| `notebooks/04_e1_weather_collisions.ipynb` | **E1**: frequency (Poisson/NB) and severity (logistic, GBT, PyTorch) | 10 min |
| `notebooks/05_e2_speed.ipynb` | **E2**: speed fixed-effects models; speed–collision NB | 10 min |
| `notebooks/06_e3_exposure.ipynb` | **E3**: cyclist collisions per trip | 10 min |
| `notebooks/07_robustness.ipynb` | robustness checks; hypotheses H1–H4 summary | 10 min |

\*on the hardware above. Run E4 from a terminal with no other heavy programs open.

Figures are written to `results/figures/` and tables to `results/tables/`.

### Notes on scope

- Detailed speed analysis and E4 use the 2023–2025 WYS stationary-sign records (144.2 million rows). The 2017–2022 archives can be added with `include_archive_years` in `src/wys.py`.
- RQ3 uses Bike Share files from 2019–2025. Part of the 2017 data has no station IDs, and the 2017–2018 files mix day/month orders.

## Reproducibility

- All parameters (paths, date ranges, buffer radii, random seed) live in `config/config.yaml`.
- Random seeds are fixed.
- Minimum versions are listed in `requirements.txt`. Hardware details are written to `results/e4/hardware.json` with the performance runs.

## AI usage

See [`docs/AI_USAGE.md`](docs/AI_USAGE.md).

## Licence and data terms

Code: MIT. Data: Open Government Licence – Toronto and Open Government Licence – Canada.
