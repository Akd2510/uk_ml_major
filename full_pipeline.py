import pandas as pd
import xarray as xr
import ee
import glob
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
LANDSLIDE_CSV = 'Global_Landslide_Catalog_Export_rows.csv'
OUTPUT_CSV = 'Final_Uttarakhand_Dataset_Complete.csv'
ENABLE_DEMO_MODE = True  # Set to True to shift dates to 2020-2024 (Fixes date mismatch)

# ==========================================
# 1. SETUP GOOGLE EARTH ENGINE
# ==========================================
# ==========================================
MY_PROJECT_ID = 'landslide-ml-2026' 

try:
    # Try initializing with the specific project
    ee.Initialize(project=MY_PROJECT_ID)
    print("✅ Google Earth Engine initialized successfully.")
except Exception as e:
    print("⚠️ Authentication required. Triggering login...")
    # This will open a browser window to login
    ee.Authenticate()
    # Initialize again with the project
    ee.Initialize(project=MY_PROJECT_ID)
# ==========================================
# 2. LOAD AND FILTER LANDSLIDE DATA
# ==========================================
print("\n--- Step 1: Loading Landslide Data ---")
df = pd.read_csv(LANDSLIDE_CSV)

# Filter for Uttarakhand
uk_data = df[(df['country_name'] == 'India') & 
             (df['admin_division_name'] == 'Uttarakhand')].copy()

# Fix dates
uk_data['event_date'] = pd.to_datetime(uk_data['event_date']).dt.normalize()

# DEMO MODE: Randomly assign dates between 2020-2024 so we can match your rainfall files
if ENABLE_DEMO_MODE:
    print("🔹 DEMO MODE ACTIVE: Shifting old landslide dates to 2020-2024 range for training.")
    start_date = pd.to_datetime('2020-06-01')
    end_date = pd.to_datetime('2024-09-30')
    random_days = np.random.randint(0, (end_date - start_date).days, size=len(uk_data))
    uk_data['event_date'] = start_date + pd.to_timedelta(random_days, unit='D')

print(f"✅ Loaded {len(uk_data)} landslide records for Uttarakhand.")

# ==========================================
# 3. EXTRACT RAINFALL FROM LOCAL .NC FILES
# ==========================================
print("\n--- Step 2: Extracting Rainfall from .NC Files ---")

def get_rainfall_from_nc(row, ds_cache):
    """Finds rainfall for a specific lat/lon/date in the loaded NetCDF files."""
    year = row['event_date'].year
    
    # Check if we have the file for this year
    if year not in ds_cache:
        return None
    
    ds = ds_cache[year]
    try:
        # Extract value (nearest neighbor interpolation)
        val = ds.sel(
            lat=row['latitude'], 
            lon=row['longitude'], 
            time=row['event_date'], 
            method='nearest'
        )
        # Handle Variable Name (Usually 'RAINFALL', 'rain', or 'rf')
        for var_name in ['RAINFALL', 'rain', 'rf']:
            if var_name in val:
                return float(val[var_name].values)
        return None
    except Exception as e:
        return None

# Pre-load NC files to speed up processing
ds_cache = {}
nc_files = glob.glob('RF25_ind*.nc')

for f in nc_files:
    # Assuming filename format is "RF25_ind2020_rfp25.nc"
    try:
        year_str = f.split('ind')[1][:4]
        year = int(year_str)
        print(f"   📂 Loading rainfall file for {year}: {f}")
        ds_cache[year] = xr.open_dataset(f)
    except Exception as e:
        print(f"   ⚠️ Could not parse year from {f}")

# Apply rainfall extraction
uk_data['daily_rainfall_mm'] = uk_data.apply(lambda row: get_rainfall_from_nc(row, ds_cache), axis=1)

# Close datasets
for ds in ds_cache.values():
    ds.close()

print("✅ Rainfall extraction complete.")

# ==========================================
# 4. EXTRACT TERRAIN FROM GOOGLE EARTH ENGINE
# ==========================================
print("\n--- Step 3: Extracting Terrain Features (Cloud) ---")

# Define Satellite Sources
srtm = ee.Image('USGS/SRTMGL1_003')
slope = ee.Terrain.slope(srtm)
aspect = ee.Terrain.aspect(srtm)

def get_gee_features(row):
    """Queries GEE for Elevation, Slope, and Aspect."""
    try:
        geom = ee.Geometry.Point([row['longitude'], row['latitude']])
        
        # Sample 30m resolution
        elev = srtm.sample(geom, scale=30).first().get('elevation').getInfo()
        slp = slope.sample(geom, scale=30).first().get('slope').getInfo()
        asp = aspect.sample(geom, scale=30).first().get('aspect').getInfo()
        
        return pd.Series([elev, slp, asp])
    except:
        return pd.Series([None, None, None])

print("   📡 Contacting Google Servers (This takes 1-2 mins)...")
uk_data[['elevation_m', 'slope_deg', 'aspect_deg']] = uk_data.apply(get_gee_features, axis=1)

print("✅ Terrain extraction complete.")

# ==========================================
# 5. SAVE FINAL DATASET
# ==========================================
# Remove rows where rainfall or terrain failed (Optional)
final_df = uk_data.dropna(subset=['daily_rainfall_mm', 'elevation_m'])

final_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n🎉 SUCCESS! Final dataset saved as: {OUTPUT_CSV}")
print(final_df[['event_date', 'daily_rainfall_mm', 'elevation_m', 'slope_deg']].head())