import pandas as pd
import xarray as xr
import ee
import os
import glob
import sys

# CONFIG
INPUT_FILE = os.path.join('outputs', 'models', 'step1_balanced_coordinates.csv')
OUTPUT_FILE = os.path.join('outputs', 'models', 'step2_final_dataset.csv')
RAINFALL_DIR = 'data'
MY_PROJECT_ID = 'landslide-ml-2026'  # <--- CHANGE THIS IF NEEDED

def init_gee():
    try:
        ee.Initialize(project=MY_PROJECT_ID)
        print("✅ GEE Initialized.")
    except Exception:
        print("⚠️ GEE Authentication required. Follow the browser prompt:")
        ee.Authenticate()
        ee.Initialize(project=MY_PROJECT_ID)

def get_terrain_features(df):
    print("📡 Extracting Terrain (Slope/Elevation) from GEE...")
    srtm = ee.Image('USGS/SRTMGL1_003')
    terrain = ee.Terrain.products(srtm)

    def extract(row):
        try:
            geom = ee.Geometry.Point([row['longitude'], row['latitude']])
            sampled = terrain.sample(geom, scale=30).first().getInfo()
            return pd.Series([sampled['properties']['elevation'], 
                              sampled['properties']['slope'], 
                              sampled['properties']['aspect']])
        except:
            return pd.Series([None, None, None])

    df[['elevation', 'slope', 'aspect']] = df.apply(extract, axis=1)
    return df

def get_rainfall_features(df):
    print("🌧️ Extracting Rainfall from .nc files...")
    nc_files = glob.glob(os.path.join(RAINFALL_DIR, '*.nc'))
    
    if not nc_files:
        print("⚠️ Warning: No .nc files found in 'data/'. Skipping rainfall.")
        df['rainfall'] = 0
        return df

    # Dictionary to cache loaded datasets (Year -> Dataset)
    datasets = {}
    
    def extract_rain(row):
        year = row['event_date'].year
        # Find file matching the year
        matching_file = next((f for f in nc_files if str(year) in f), None)
        
        if not matching_file: return 0.0
        
        if year not in datasets:
            datasets[year] = xr.open_dataset(matching_file)
            
        try:
            val = datasets[year].sel(lat=row['latitude'], lon=row['longitude'], 
                                     time=row['event_date'], method='nearest')
            # Adjust 'RAINFALL' if your variable name is different (e.g., 'rain')
            return float(val['RAINFALL'].values)
        except:
            return 0.0

    df['event_date'] = pd.to_datetime(df['event_date'])
    df['rainfall'] = df.apply(extract_rain, axis=1)
    
    # Close datasets
    for ds in datasets.values(): ds.close()
    
    return df

def main():
    if not os.path.exists(INPUT_FILE):
        print("❌ Run '1_preprocess.py' first.")
        return

    df = pd.read_csv(INPUT_FILE)
    init_gee()
    
    df = get_terrain_features(df)
    df = get_rainfall_features(df)
    
    # Remove rows where features failed
    df_clean = df.dropna()
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Final dataset ready: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()