import pandas as pd
from pathlib import Path

RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')

# We'll use sklearn's fetch_openml for Ames Housing to avoid manual download.
from sklearn.datasets import fetch_openml

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    data = fetch_openml(name='house_prices', as_frame=True, parser='auto')
    df = data.frame
    # Save raw
    df.to_csv(RAW_DIR / 'ames_raw.csv', index=False)
    # Minimal clean: drop rows with all NA saleprice
    df = df[df['SalePrice'].notna()].copy()
    df.to_csv(PROCESSED_DIR / 'ames_clean.csv', index=False)
    print('Downloaded and prepared data:', df.shape)

if __name__ == '__main__':
    main()
