import pandas as pd
import xarray as xr
import os
import glob

# 1. LOAD AND FILTER LANDSLIDE DATA
landslide_file = 'Global_Landslide_Catalog_Export_rows.csv'
df = pd.read_csv(landslide_file)

# Filter for Uttarakhand
uk_data = df[(df['country_name'] == 'India') & 
             (df['admin_division_name'] == 'Uttarakhand')].copy()
uk_data['event_date'] = pd.to_datetime(uk_data['event_date']).dt.normalize()

# 2. DEFINE RAINFALL EXTRACTION FUNCTION
def get_rainfall_for_event(row, ds):
    """Extracts rainfall value for a specific lat, lon, and date."""
    try:
        # Match nearest lat/lon and exact date
        rain_val = ds.sel(
            lat=row['latitude'], 
            lon=row['longitude'], 
            time=row['event_date'], 
            method='nearest'
        )
        # Replace 'RAINFALL' with the actual variable name in your .nc files (usually 'rain' or 'rf')
        return float(rain_val.RAINFALL.values) 
    except Exception:
        return None

# 3. PROCESS EACH YEARLY RAINFALL FILE
# List of your uploaded .nc files
rainfall_files = glob.glob('RF25_ind*.nc')
all_matched_data = []

print("Processing rainfall files...")

for file in rainfall_files:
    print(f"Opening {file}...")
    ds = xr.open_dataset(file)
    
    # Check for landslides that occurred in the year of the current file
    year = int(file.split('_ind')[1][:4])
    year_landslides = uk_data[uk_data['event_date'].dt.year == year].copy()
    
    if not year_landslides.empty:
        print(f"Found {len(year_landslides)} events in {year}. Extracting rainfall...")
        year_landslides['daily_rainfall_mm'] = year_landslides.apply(
            get_rainfall_for_event, axis=1, ds=ds
        )
        all_matched_data.append(year_landslides)
    
    ds.close()

# 4. CONSOLIDATE AND SAVE
if all_matched_data:
    final_df = pd.concat(all_matched_data)
    final_df.to_csv('Uttarakhand_Landslides_with_Rainfall.csv', index=False)
    print(f"Success! Saved merged data to 'Uttarakhand_Landslides_with_Rainfall.csv'")
else:
    print("No date overlap found between landslide records and rainfall files.")
    # Saving the empty-rainfall version for model structure purposes
    uk_data.to_csv('Uttarakhand_Landslide_Template.csv', index=False)