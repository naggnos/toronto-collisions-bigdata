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
├── config/config.yaml        # paths, date ranges, buffer radii, seeds
├── data/                     # raw and processed data (not committed, see data/README.md)
├── notebooks/                # numbered notebooks, run in order
├── src/                      # shared helpers (Spark session, config loader)
├── results/
│   ├── figures/              # figures used in the report
│   └── tables/               # result tables (CSV)
├── docs/
│   ├── CONTRIBUTIONS.md      # task distribution between team members
│   └── AI_USAGE.md           # AI tool disclosure
└── requirements.txt
```

## Datasets

All datasets are open data. Raw files are not stored in this repository because of their size (about 17 GB). See [`data/README.md`](data/README.md) for download links and where to place the files.

| ID | Dataset | Publisher |
|---|---|---|
| D1 | Traffic Collisions | Toronto Police Service |
| D2 | Killed or Seriously Injured (KSI) collisions | Toronto Police Service |
| D3 | Watch Your Speed – school-zone and mobile sign speed counts | City of Toronto |
| D4 | Bike Share Toronto ridership | Toronto Parking Authority |
| D5 | Hourly weather, three Toronto stations | ECCC |

## How to run

### 1. Requirements

- Python 3.11+
- Java 17 (required by Spark)
- About 60 GB free disk space for raw data, Parquet copies and intermediate tables

### 2. Set up the environment

```bash
git clone https://github.com/naggnos/toronto-collisions-bigdata.git
cd toronto-collisions-bigdata
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get the data

Download the files listed in `data/README.md` into `data/raw/`, following the folder names given there.

### 4. Run the notebooks in order

| Notebook | Purpose |
|---|---|
| `00_manifest_and_parquet.ipynb` | Record every raw file (URL, size, SHA-256); convert to Parquet partitioned by year/month |
| `01_data_quality_eda.ipynb` | Data-quality checks (missing coordinates, duplicates, severity definition, station coverage, time zones, COVID break) and EDA |
| `02_e4_tool_comparison.ipynb` | E4: same WYS aggregation + join workload in pandas, DuckDB and Spark; selects the processing tool |
| `03_pipeline.ipynb` | Composite weather table; WYS sign-hour metrics; city-hour and neighbourhood-hour panels; grid-based spatial join of collisions to WYS signs |
| `04_e1_weather_collisions.ipynb` | E1 / RQ1: frequency (negative binomial) and severity (logistic regression, GBT) models |
| `05_e2_speed.ipynb` | E2 / RQ2: speed adaptation and speed–collision models |
| `06_e3_exposure.ipynb` | E3 / RQ3: cyclist risk per Bike Share trip |
| `07_robustness.ipynb` | Robustness checks for E1–E3 |

Figures and tables are written to `results/`.

## Reproducibility

- All parameters (paths, date ranges, buffer radii, random seed) live in `config/config.yaml`.
- Random seeds are fixed.
- Software versions are pinned in `requirements.txt`. Hardware details are logged with each performance run.

## AI usage

See [`docs/AI_USAGE.md`](docs/AI_USAGE.md).

## Licence and data terms

Code: MIT. Data: Open Government Licence – Toronto and Open Government Licence – Canada.
