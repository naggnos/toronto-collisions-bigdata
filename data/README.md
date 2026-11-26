# Data

Raw data are not committed (about 17 GB). Download each dataset into `data/raw/<folder>/` as shown below. All sources are free and need no registration.

| Folder | Dataset | Source |
|---|---|---|
| `collisions/` | D1 Traffic Collisions (Toronto Police) | https://open.toronto.ca/dataset/police-annual-statistical-report-traffic-collisions/ |
| `ksi/` | D2 Motor Vehicle Collisions involving KSI persons | https://open.toronto.ca/dataset/motor-vehicle-collisions-involving-killed-or-seriously-injured-persons/ |
| `wys_school/` | D3 School Safety Zone WYS – detailed speed counts and locations | https://open.toronto.ca/dataset/school-safety-zone-watch-your-speed-program-detailed-speed-counts/ |
| `wys_mobile/` | D3 Mobile WYS – detailed speed counts | https://open.toronto.ca/dataset/mobile-watch-your-speed-program-detailed-speed-counts/ |
| `bikeshare/` | D4 Bike Share Toronto ridership (optional) | https://open.toronto.ca/dataset/bike-share-toronto-ridership-data/ |
| `weather/` | D5 ECCC hourly data: Toronto Intl A, Toronto City, Toronto City Centre (2014–2025) | https://climate.weather.gc.ca/historical_data/search_historic_data_e.html |
| `boundaries/` | Neighbourhoods (158) and Toronto Centreline | https://open.toronto.ca/dataset/neighbourhoods/ |

`manifest.csv` (to be added) records the source URL, file name, size and SHA-256 checksum of every raw file used.

Processed Parquet tables are written to `data/processed/`, partitioned by year and month.
