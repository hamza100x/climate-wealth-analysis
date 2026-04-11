"""
data_loader.py
Fetches and merges GDP, temperature, and latitude data.
"""

import os
import numpy as np
import pandas as pd


def _find_col(columns, candidates):
    """Return first matching column name using case-insensitive candidates."""
    lowered = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        found = lowered.get(candidate.lower().strip())
        if found:
            return found
    return None


def fetch_gdp(filepath: str = 'data/gdp-per-capita-worldbank.csv', year: int = 2019) -> pd.DataFrame:
    """Load GDP per capita data and normalize to country_name, iso3, gdp_ppp."""
    print("  Fetching GDP data...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"GDP file not found at '{filepath}'. Expected a local CSV with GDP per capita values."
        )

    df = pd.read_csv(filepath)

    country_col = _find_col(df.columns, ['country_name', 'country', 'entity', 'name'])
    iso3_col = _find_col(df.columns, ['iso3', 'code', 'economy', 'alpha-3', 'alpha3'])
    year_col = _find_col(df.columns, ['year', 'time'])
    gdp_col = _find_col(
        df.columns,
        ['gdp_ppp', 'gdp per capita', f'YR{year}', str(year), 'value']
    )

    if gdp_col is None:
        for col in df.columns:
            name = col.lower().strip()
            if 'gdp' in name and 'capita' in name:
                gdp_col = col
                break

    missing = []
    if country_col is None:
        missing.append('country')
    if iso3_col is None:
        missing.append('iso3/code')
    if gdp_col is None:
        missing.append('gdp value')
    if missing:
        raise ValueError(f"Could not map GDP columns: missing {', '.join(missing)}")

    out = df.rename(columns={
        country_col: 'country_name',
        iso3_col: 'iso3',
        gdp_col: 'gdp_ppp',
    })

    out['gdp_ppp'] = pd.to_numeric(out['gdp_ppp'], errors='coerce')

    if year_col is not None:
        out = out.rename(columns={year_col: 'year'})
        out['year'] = pd.to_numeric(out['year'], errors='coerce')
        if (out['year'] == year).any():
            out = out[out['year'] == year]
        else:
            out = out.sort_values('year').groupby('iso3', as_index=False).tail(1)

    return out[['country_name', 'iso3', 'gdp_ppp']].dropna(subset=['gdp_ppp'])


def fetch_latitude(filepath: str = 'data/countries-with-latitude.csv') -> pd.DataFrame:
    """Load country coordinates from local CSV, fallback to public CSV."""
    print("  Fetching coordinates...")
    if os.path.exists(filepath):
        local = pd.read_csv(filepath)
        country_col = _find_col(local.columns, ['country_name', 'country', 'entity', 'name'])
        lat_col = _find_col(local.columns, ['latitude', 'lat'])
        lon_col = _find_col(local.columns, ['longitude', 'long', 'lon'])

        if country_col and lat_col and lon_col:
            out = local.rename(columns={
                country_col: 'country_name',
                lat_col: 'latitude',
                lon_col: 'longitude',
            })
            return out[['country_name', 'latitude', 'longitude']]

    url = (
        "https://raw.githubusercontent.com/albertyw/"
        "avenews/master/old/data/average-latitude-longitude-countries.csv"
    )
    df = pd.read_csv(url, usecols=['Country', 'Latitude', 'Longitude'])
    return df.rename(columns={
        'Country': 'country_name',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
    })


def fetch_region_codes(filepath: str = 'data/countries-with-latitude.csv') -> pd.DataFrame:
    """Load ISO3 country names with region and sub-region labels."""
    print("  Fetching region codes...")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        url = (
            "https://raw.githubusercontent.com/lukes/"
            "ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"
        )
        df = pd.read_csv(url)

    iso3_col = _find_col(df.columns, ['iso3', 'alpha-3', 'code'])
    name_col = _find_col(df.columns, ['country_name', 'country', 'entity', 'name'])
    region_col = _find_col(df.columns, ['region'])
    sub_region_col = _find_col(df.columns, ['sub-region', 'sub_region'])

    if not all([iso3_col, name_col, region_col, sub_region_col]):
        raise ValueError('Region codes file is missing one or more required columns.')

    return df.rename(columns={
        iso3_col: 'iso3',
        name_col: 'country_name',
        region_col: 'region',
        sub_region_col: 'sub-region',
    })[['country_name', 'iso3', 'region', 'sub-region']]


def load_temperature(filepath: str = 'data/average-monthly-surface-temperature.csv') -> pd.DataFrame:
    """
    Load temperature CSV downloaded from World Bank Climate Portal.
    https://climateknowledgeportal.worldbank.org/download-data
    Select: Historical → Country averages → Temperature → 1991-2020 → Annual
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\nTemperature file not found at '{filepath}'.\n"
            "Please download it from:\n"
            "  https://climateknowledgeportal.worldbank.org/download-data\n"
            "Select: Historical → Country averages → Temperature → 1991-2020 → Annual\n"
            "Save the file as: data/temperature.csv"
        )

    df = pd.read_csv(filepath)
    print(f"  Temperature file columns: {list(df.columns)}")

    country_col = _find_col(df.columns, ['country_name', 'country', 'entity', 'name'])
    iso3_col = _find_col(df.columns, ['iso3', 'code', 'alpha-3'])
    temp_col = _find_col(df.columns, ['mean_temp', 'monthly average', 'temperature', 'temp'])

    if country_col is None or temp_col is None:
        raise ValueError('Could not detect temperature country/value columns.')

    out = df.rename(columns={
        country_col: 'country_name',
        temp_col: 'mean_temp',
    })
    out['mean_temp'] = pd.to_numeric(out['mean_temp'], errors='coerce')

    group_keys = ['country_name']
    if iso3_col is not None:
        out = out.rename(columns={iso3_col: 'iso3'})
        group_keys = ['country_name', 'iso3']

    out = out.groupby(group_keys, as_index=False)['mean_temp'].mean()
    return out


def build_dataset(temp_filepath: str = 'data/average-monthly-surface-temperature.csv') -> pd.DataFrame:
    """Fetch all sources, merge, engineer features. Returns analysis-ready DataFrame."""
    print("\nBuilding dataset...")

    gdp = fetch_gdp()
    temp = load_temperature(temp_filepath)
    coords = fetch_latitude()
    regions = fetch_region_codes()

    # Prefer ISO3 for core merge to avoid country-name spelling mismatches.
    if 'iso3' in temp.columns:
        df = gdp.merge(temp[['iso3', 'mean_temp']], on='iso3', how='inner')
    else:
        df = gdp.merge(temp[['country_name', 'mean_temp']], on='country_name', how='inner')

    df = df.merge(coords, on='country_name', how='left')
    df = df.merge(
        regions[['iso3', 'region', 'sub-region']],
        on='iso3', how='left'
    )

    # Feature engineering
    df['log_gdp']      = np.log(df['gdp_ppp'])
    df['abs_latitude'] = df['latitude'].abs()

    # Drop rows missing any key variable
    before = len(df)
    df = df.dropna(subset=['log_gdp', 'mean_temp', 'abs_latitude'])
    print(f"\n  Dropped {before - len(df)} rows with missing values")
    print(f"  Final dataset: {len(df)} countries, {df.shape[1]} columns")

    return df.reset_index(drop=True)