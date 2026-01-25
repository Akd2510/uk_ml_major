import pandas as pd
import os

# 1. SETUP
csv_file = 'Global_Landslide_Catalog_Export_rows.csv'

# 2. DEFINE THE NEW DATA (2020-2024 Events)
# These match your rainfall .nc files perfectly
new_events = [
    {'event_date': '2021-02-07', 'latitude': 30.4166, 'longitude': 79.6666, 'country_name': 'India', 'admin_division_name': 'Uttarakhand', 'event_title': 'Chamoli Disaster'},
    {'event_date': '2021-10-19', 'latitude': 29.3800, 'longitude': 79.4500, 'country_name': 'India', 'admin_division_name': 'Uttarakhand', 'event_title': 'Nainital Cloudburst'},
    {'event_date': '2022-08-20', 'latitude': 30.3165, 'longitude': 78.0322, 'country_name': 'India', 'admin_division_name': 'Uttarakhand', 'event_title': 'Maldevta Flashflood'},
    {'event_date': '2023-08-04', 'latitude': 30.5500, 'longitude': 79.0333, 'country_name': 'India', 'admin_division_name': 'Uttarakhand', 'event_title': 'Gaurikund Landslide'},
    {'event_date': '2023-07-11', 'latitude': 30.7400, 'longitude': 78.4300, 'country_name': 'India', 'admin_division_name': 'Uttarakhand', 'event_title': 'Jumma Rain Disaster'}
]

# 3. LOAD EXISTING FILE
try:
    df = pd.read_csv(csv_file)
    print(f"Original Row Count: {len(df)}")
    
    # 4. APPEND NEW ROWS
    new_df = pd.DataFrame(new_events)
    # Align columns (fill missing columns with NaN)
    df_updated = pd.concat([df, new_df], ignore_index=True)
    
    # 5. SAVE BACK TO CSV
    df_updated.to_csv(csv_file, index=False)
    
    print("------------------------------------------------")
    print(f"✅ SUCCESS! Added 5 recent events to {csv_file}")
    print(f"New Row Count: {len(df_updated)}")
    print("------------------------------------------------")
    print("You can now run 'full_pipeline.py' with DEMO_MODE = False.")

except Exception as e:
    print(f"❌ Error: {e}")