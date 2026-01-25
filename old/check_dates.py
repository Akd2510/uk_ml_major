import pandas as pd

# 1. Load the CSV
csv_file = 'Global_Landslide_Catalog_Export_rows.csv'
print(f"Reading {csv_file}...")

try:
    df = pd.read_csv(csv_file)
    
    # 2. Convert Date Column
    # errors='coerce' turns bad dates into NaT (Not a Time) so the script doesn't crash
    df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    
    # 3. Filter for Recent Events (2020-2024)
    recent_events = df[ (df['event_date'].dt.year >= 2020) & (df['event_date'].dt.year <= 2024) ]
    
    print("\n---------------------------------------------------")
    print(f"🔍 ANALYSIS REPORT")
    print("---------------------------------------------------")
    
    if not recent_events.empty:
        count = len(recent_events)
        print(f"✅ SUCCESS: Found {count} events between 2020 and 2024.")
        print("Your pipeline WILL work without Demo Mode.")
        print("\nHere are the events the model will use:")
        print(recent_events[['event_date', 'latitude', 'longitude']].head())
    else:
        print("❌ FAILURE: No events found between 2020-2024.")
        print("Possible reasons:")
        print("1. You didn't save the CSV file after adding the rows.")
        print("2. The date format is wrong (Use YYYY-MM-DD, e.g., 2023-07-11).")
        print("3. You added them to the wrong file.")

except Exception as e:
    print(f"❌ ERROR reading file: {e}")