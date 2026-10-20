import sys
from pathlib import Path
from string import Template

import duckdb

import config

# every placeholder in sql/*.sql comes from here: ${listing_path}, ${PRICE_MIN}, ...
PARAMS = {k: v for k, v in vars(config).items() if k.isupper() or k.endswith('_PATH')}
PARAMS.update(listing_path=config.LISTING_PATH, neighbor_path=config.NEIGHBORHOOD_PATH, price_max=config.PRICE_MAX, price_min=config.PRICE_MIN, features_path=config.FEATURES_PATH)

if __name__ == '__main__':
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(config.DB_PATH)) as con:
        for f in sys.argv[1:]:
            con.execute(Template(Path(f).read_text()).substitute(PARAMS))
            print(f'ran {f}')
