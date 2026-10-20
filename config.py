SEED=42
SNAPSHOT_DATE="2026-09-22"
BASE_URL = "https://data.insideairbnb.com/thailand/central-thailand/bangkok/"
LISTING_URL = BASE_URL + SNAPSHOT_DATE + "/data/listings.csv.gz"
NEIGHBORHOOD_URL = BASE_URL + SNAPSHOT_DATE + "/visualisations/neighbourhoods.csv"
NEIGHBORHOOD_GEO_URL = BASE_URL + SNAPSHOT_DATE + "/visualisations/neighbourhoods.geojson"
TEST_SIZE = 0.2
N_FOLDS = 5

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DB_PATH = ROOT / "airbnb.duckdb"
SQL_DIR = ROOT / "sql"

LISTING_PATH = RAW_DIR / "listings.csv.gz"
NEIGHBORHOOD_PATH = RAW_DIR / "neighbourhoods.csv"
NEIGHBORHOOD_GEO_PATH = RAW_DIR / "neighbourhoods.geojson"

PRICE_MIN = 361
PRICE_MAX = 18541
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
SPLIT_PATH = PROCESSED_DIR / "split.parquet"

RESULTS_DIR = ROOT / "results"
REF_LAT = 13.7456  # Siam BTS, city centre reference for dist_to_center_km
REF_LON = 100.5341
