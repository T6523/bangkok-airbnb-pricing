from pathlib import Path
from urllib.request import urlretrieve

from config import LISTING_URL, NEIGHBORHOOD_URL, NEIGHBORHOOD_GEO_URL

RAW = Path('data/raw')
FILES = {
    'listings.csv.gz': LISTING_URL,
    'neighbourhoods.csv': NEIGHBORHOOD_URL,
    'neighbourhoods.geojson': NEIGHBORHOOD_GEO_URL,
}

if __name__ == '__main__':
    RAW.mkdir(parents=True, exist_ok=True)
    
    for name, url in FILES.items():
        path = RAW / name
        if path.exists():
            print(f'skip {path} (exists)')
            continue
        urlretrieve(url, path)
        print(f'file saved at {path}')
