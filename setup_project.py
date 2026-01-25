import os
import sys
import subprocess

def setup_environment():
    print("--- 🛠️ Setting up Project Environment ---")

    # 1. Create Directories
    dirs = ['data', 'outputs/figures', 'outputs/models']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✅ Directory ready: {d}")

    # 2. Check Data Files
    required_csv = os.path.join('data', 'Global_Landslide_Catalog_Export_rows.csv')
    if not os.path.exists(required_csv):
        print(f"❌ CRITICAL ERROR: Move 'Global_Landslide_Catalog_Export_rows.csv' into the 'data/' folder!")
    else:
        print(f"✅ Found Landslide Catalog.")

    print("\n--- Setup Complete. You may now run '1_preprocess.py' ---")

if __name__ == "__main__":
    setup_environment()