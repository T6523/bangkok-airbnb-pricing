from pathlib import Path
from urllib.request import urlretrieve

from config import LISTING_URL, NEIGHBORHOOD_URL, NEIGHBORHOOD_GEO_URL , LISTING_PATH, NEIGHBORHOOD_PATH, NEIGHBORHOOD_GEO_PATH

RAW = Path('data/raw')
FILES = {
    LISTING_PATH : LISTING_URL,
    NEIGHBORHOOD_PATH : NEIGHBORHOOD_URL,
    NEIGHBORHOOD_GEO_PATH : NEIGHBORHOOD_GEO_URL,
}

if __name__ == '__main__':
    
    for path, url in FILES.items():
        if path.exists():
            print(f'skip {path} (exists)')
            continue
        urlretrieve(url, path)
        print(f'file saved at {path}')