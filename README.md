# Weather, Operating Speed and Exposure: Toronto Motor-Vehicle Collisions, 2014–2025

CSCI 6886 Big Data Analytics (Dr. Salma Noor), Fairleigh Dickinson University Vancouver
Team: **Gan Song** (2095706) and **Mingye Lang** (2099150)

This repository holds the implementation for Deliverable 2 (Implementation and Evaluation) of our project. We integrate Toronto Police collision records, City of Toronto Watch Your Speed (WYS) sign counts and Environment and Climate Change Canada (ECCC) hourly weather. Then we use Apache Spark (PySpark, local mode) to study how weather is associated with collision frequency and severity, and whether measured vehicle speeds change in adverse weather.

## Research questions (Deliverable 2 scope)

| | Question | Method |
|---|---|---|
| **Main (RQ1)** | After controlling for hour, day of week, season, year and the COVID-19 period, how are rain, snow, low visibility and low temperature associated with collisions per hour and with the probability that a collision is severe (injury or fatal)? | Negative binomial regression (frequency); PyTorch logistic regression (severity) |
| **Supporting (RQ2)** | Do 85th-percentile speeds and the share of speeding vehicles measured by WYS signs decrease in adverse weather? | Linear regression with sign fixed effects |

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
| D4 | Bike Share Toronto ridership (optional, RQ3) | Toronto Parking Authority |
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

| Notebook | Purpose | Report section |
|---|---|---|
| `01_data_preparation_eda.ipynb` | Load, clean, deduplicate, EDA | §2 |
| `02_spark_processing.ipynb` | Spark aggregation and joins: WYS sign-hour metrics, city-hour collision/weather table | §3 |
| `03_severity_model_pytorch.ipynb` | PyTorch severity model: training, time-based split, evaluation | §4 |
| `04_experiments_rq1_rq2.ipynb` | RQ1 negative binomial model and RQ2 speed model | §5 |
| `05_performance_evaluation.ipynb` | Spark execution time on 50% vs 100% of the data | §6 |

Figures and tables are written to `results/`.

## Reproducibility

- All parameters (paths, date ranges, buffer radii, random seed) live in `config/config.yaml`.
- Random seeds are fixed.
- Software versions are pinned in `requirements.txt`. Hardware details are logged with each performance run.

## AI usage

See [`docs/AI_USAGE.md`](docs/AI_USAGE.md).

## Licence and data terms

Code: MIT. Data: Open Government Licence – Toronto and Open Government Licence – Canada.
