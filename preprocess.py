import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_FILE  = os.path.join('data', 'Global_Landslide_Catalog_Export_rows.csv')
OUTPUT_FILE = os.path.join('outputs', 'models', 'step1_balanced_coordinates.csv')

# Uttarakhand bounding box (used for random negative sampling)
UK_LAT_MIN, UK_LAT_MAX = 28.7, 31.5
UK_LON_MIN, UK_LON_MAX = 77.5, 81.1

# Negative sampling settings
NEGATIVE_RATIO   = 5      # 5 negatives per positive — enough signal without flooding
BUFFER_DEG       = 0.05   # ~5 km exclusion ring around each known landslide (~0.05°)
RANDOM_SEED      = 42


def preprocess():
    print("--- Step 1: Pre-processing & Negative Sampling ---")

    # ── 1. Load ────────────────────────────────────────────
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ Error: Input CSV not found in 'data/' folder.")
        return

    # ── 2. Filter for Uttarakhand ──────────────────────────
    uk_data = df[
        (df['country_name'] == 'India') &
        (df['admin_division_name'] == 'Uttarakhand')
    ].copy()

    uk_data['event_date'] = pd.to_datetime(uk_data['event_date'], errors='coerce')
    uk_data = uk_data.dropna(subset=['event_date', 'latitude', 'longitude'])

    n_pos = len(uk_data)
    print(f"✅ Found {n_pos} confirmed landslide events in Uttarakhand.")

    if n_pos == 0:
        print("❌ No positive samples found. Check your CSV filters.")
        return

    # ── 3. Proper negative sampling ────────────────────────
    #
    # OLD approach (yours): jitter positive coordinates by ±0.1°.
    # PROBLEM: negatives land 5–10 km from real landslides — still
    # geologically similar terrain. The model learns almost nothing
    # about genuinely safe zones, so it over-predicts risk everywhere.
    #
    # NEW approach: randomly sample coordinates from the full
    # Uttarakhand bounding box, then discard any that fall within
    # BUFFER_DEG of a known landslide. Pair each negative with a
    # random date from the positive catalog so rainfall can be matched.
    #
    rng         = np.random.default_rng(RANDOM_SEED)
    pos_coords  = uk_data[['latitude', 'longitude']].values
    event_dates = uk_data['event_date'].values

    negatives   = []
    attempts    = 0
    max_attempts = NEGATIVE_RATIO * n_pos * 50  # safety cap

    while len(negatives) < NEGATIVE_RATIO * n_pos and attempts < max_attempts:
        attempts += 1
        lat = rng.uniform(UK_LAT_MIN, UK_LAT_MAX)
        lon = rng.uniform(UK_LON_MIN, UK_LON_MAX)

        # Reject if too close to any known landslide
        dists = np.sqrt((pos_coords[:, 0] - lat)**2 + (pos_coords[:, 1] - lon)**2)
        if dists.min() < BUFFER_DEG:
            continue

        # Borrow a random event date so the rainfall extractor can work
        date = rng.choice(event_dates)
        negatives.append({'event_date': date, 'latitude': lat, 'longitude': lon, 'label': 0})

    neg_df = pd.DataFrame(negatives)
    n_neg  = len(neg_df)
    print(f"✅ Generated {n_neg} negative samples (ratio 1:{n_neg // n_pos}) "
          f"with {BUFFER_DEG}° ({BUFFER_DEG * 111:.0f} km) exclusion buffer.")

    if n_neg < NEGATIVE_RATIO * n_pos * 0.8:
        print("⚠️  Warning: fewer negatives than expected — bounding box may be too tight "
              "or BUFFER_DEG too large. Consider reducing BUFFER_DEG.")

    # ── 4. Combine & shuffle ───────────────────────────────
    uk_data['label'] = 1
    balanced_df = pd.concat([
        uk_data[['event_date', 'latitude', 'longitude', 'label']],
        neg_df
    ]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # ── 5. Save ────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    balanced_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved balanced dataset ({len(balanced_df)} rows) → {OUTPUT_FILE}")
    print(f"   Class distribution:\n{balanced_df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    preprocess()