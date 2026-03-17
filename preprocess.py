import pandas as pd
import numpy as np
import os

# CONFIG
INPUT_FILE = os.path.join('data', 'Global_Landslide_Catalog_Export_rows.csv')
OUTPUT_FILE = os.path.join('outputs', 'models', 'step1_balanced_coordinates.csv')

def preprocess():
    print("--- Step 1: Pre-processing & Balancing ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ Error: Input CSV not found in 'data/' folder.")
        return

    # 2. Filter for Uttarakhand
    uk_data = df[(df['country_name'] == 'India') & 
                 (df['admin_division_name'] == 'Uttarakhand')].copy()
    
    # 3. Standardize Dates
    uk_data['event_date'] = pd.to_datetime(uk_data['event_date'], errors='coerce')
    uk_data = uk_data.dropna(subset=['event_date', 'latitude', 'longitude'])
    
    print(f"✅ Found {len(uk_data)} actual landslide events in Uttarakhand.")

    # 4. Generate Non-Landslide Samples (Negative Sampling)
    # We create random points slightly offset from real ones to represent 'safe' areas
    non_landslides = uk_data.copy()
    non_landslides['latitude'] += np.random.uniform(-0.1, 0.1, len(non_landslides))
    non_landslides['longitude'] += np.random.uniform(-0.1, 0.1, len(non_landslides))
    
    # Assign Labels: 1 = Landslide, 0 = No Landslide
    uk_data['label'] = 1
    non_landslides['label'] = 0
    
    # Combine
    balanced_df = pd.concat([uk_data, non_landslides]).sample(frac=1).reset_index(drop=True)
    
    # 5. Save
    balanced_df[['event_date', 'latitude', 'longitude', 'label']].to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved balanced dataset ({len(balanced_df)} rows) to: {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess()