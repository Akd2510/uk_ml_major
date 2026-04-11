import pandas as pd
import numpy as np
import netCDF4 as nc
import ee
import os
import glob

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_FILE    = os.path.join('outputs', 'models', 'step1_balanced_coordinates.csv')
OUTPUT_FILE   = os.path.join('outputs', 'models', 'step2_final_dataset.csv')
RAINFALL_DIR  = 'data'
MY_PROJECT_ID = 'landslide-ml-2026'

# Rainfall .nc files cover 2008–2017
# Events outside this range will get rainfall = 0
RAINFALL_YEAR_MIN = 2008
RAINFALL_YEAR_MAX = 2017


# ─────────────────────────────────────────────
# GEE INIT
# ─────────────────────────────────────────────
def init_gee():
    try:
        ee.Initialize(project=MY_PROJECT_ID)
        print("✅ GEE Initialized.")
    except Exception:
        print("⚠️  GEE Authentication required. Follow the browser prompt:")
        ee.Authenticate()
        ee.Initialize(project=MY_PROJECT_ID)


# ─────────────────────────────────────────────
# TERRAIN FEATURES — single GEE API call
# ─────────────────────────────────────────────
def get_terrain_features(df):
    """
    Extracts 7 terrain features in a SINGLE GEE call using
    ee.Image.sampleRegions() — sends all points at once instead
    of one API call per point. Much faster and avoids rate limits.

    Features extracted:
      elevation  — height above sea level (m)
      slope      — slope angle in degrees
      aspect     — compass direction of slope face
      twi        — Topographic Wetness Index (flow accumulation / tan(slope))
      curvature  — plan curvature (convergent hollows vs convex ridges)
      tpi        — Topographic Position Index (ridges vs valleys)
      ndvi       — vegetation density (deforested = higher risk)
    """
    print("📡 Extracting terrain features from GEE (single batch call)...")

    # ── Build feature image ───────────────────────────────
    srtm      = ee.Image('USGS/SRTMGL1_003')
    terrain   = ee.Terrain.products(srtm)
    elev      = srtm.rename('elevation')

    # TWI = ln(flow_accum / tan(slope))
    slope_rad  = terrain.select('slope').multiply(np.pi / 180)
    tan_slope  = slope_rad.tan().max(ee.Image(0.001))   # floor to avoid ln(0)
    flow_accum = (ee.Image('WWF/HydroSHEDS/15ACC')
                  .select('b1')
                  .rename('flow_accum'))
    twi = flow_accum.divide(tan_slope).log().rename('twi')

    # Curvature via Laplacian
    curvature = elev.convolve(
        ee.Kernel.laplacian8(normalize=False)
    ).rename('curvature')

    # TPI = elevation minus neighbourhood mean (300 m radius)
    mean_elev = elev.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.circle(radius=10, units='pixels')
    )
    tpi = elev.subtract(mean_elev).rename('tpi')

    # NDVI from Landsat 8 median composite
    ls8  = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterDate('2015-01-01', '2023-12-31')
            .filter(ee.Filter.lt('CLOUD_COVER', 20))
            .median())
    nir  = ls8.select('SR_B5').multiply(0.0000275).add(-0.2)
    red  = ls8.select('SR_B4').multiply(0.0000275).add(-0.2)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('ndvi')

    # Stack all bands into one image
    image = (terrain.select(['elevation', 'slope', 'aspect'])
             .addBands(twi)
             .addBands(curvature)
             .addBands(tpi)
             .addBands(ndvi))

    # ── Build GEE FeatureCollection from all points ───────
    print(f"   Sending {len(df)} points to GEE in one call...")

    features = []
    for idx, row in df.iterrows():
        feat = ee.Feature(
            ee.Geometry.Point([row['longitude'], row['latitude']]),
            {'row_id': idx}
        )
        features.append(feat)

    fc = ee.FeatureCollection(features)

    # ── Single sampleRegions call ─────────────────────────
    sampled = image.sampleRegions(
        collection=fc,
        scale=30,
        geometries=False
    )

    result = sampled.getInfo()
    print(f"   GEE returned {len(result['features'])} sampled points.")

    # ── Parse results back into dataframe ─────────────────
    # Build a lookup: row_id -> properties
    id_to_props = {}
    for feat in result['features']:
        props  = feat['properties']
        row_id = int(props.get('row_id', -1))
        if row_id >= 0:
            id_to_props[row_id] = props

    terrain_cols = ['elevation', 'slope', 'aspect',
                    'twi', 'curvature', 'tpi', 'ndvi']

    rows = []
    for idx in df.index:
        props = id_to_props.get(idx, {})
        rows.append({col: props.get(col, np.nan) for col in terrain_cols})

    terrain_df = pd.DataFrame(rows, index=df.index)
    df = pd.concat([df, terrain_df], axis=1)

    nan_count = df[terrain_cols].isna().any(axis=1).sum()
    print(f"✅ Terrain extraction complete.")
    print(f"   NaN rows (GEE failed to sample): {nan_count} / {len(df)}")
    if nan_count > 0:
        print(f"   These will be dropped in the final clean step.")

    return df


# ─────────────────────────────────────────────
# RAINFALL FEATURES — pure netCDF4, 2008–2017
# ─────────────────────────────────────────────
def get_rainfall_features(df):
    """
    Extracts from IMD RF25 gridded rainfall NetCDF files (2008–2017):
      rainfall        — mm on the event day
      rainfall_3d_cum — cumulative mm over the 3 days before the event

    ROOT CAUSE OF PREVIOUS 0s:
      np.abs(time_dates - target_date) fails silently when time_dates is
      a numpy array of pd.Timestamp objects — numpy cannot subtract a
      Timestamp from such an array and the except block catches it,
      returning 0.0 every time.

    FIX:
      Convert time_dates to numpy int64 (nanoseconds) and target_date
      to the same — then np.argmin works correctly and reliably.
    """
    print("🌧️  Extracting rainfall from .nc files (2008–2017)...")

    nc_files = glob.glob(os.path.join(RAINFALL_DIR, '*.nc'))
    nc_files = [f for f in nc_files if any(
        str(y) in os.path.basename(f)
        for y in range(RAINFALL_YEAR_MIN, RAINFALL_YEAR_MAX + 1)
    )]

    if not nc_files:
        print(f"⚠️  No matching .nc files found in '{RAINFALL_DIR}/'.")
        df['rainfall']        = 0.0
        df['rainfall_3d_cum'] = 0.0
        return df

    print(f"   Found {len(nc_files)} .nc files: "
          f"{sorted([os.path.basename(f) for f in nc_files])}")

    # ── Cache datasets by year ────────────────────────────
    dataset_cache = {}

    def load_year(year):
        if year in dataset_cache:
            return dataset_cache[year]
        if year < RAINFALL_YEAR_MIN or year > RAINFALL_YEAR_MAX:
            return None
        match = next(
            (f for f in nc_files if str(year) in os.path.basename(f)), None)
        if match is None:
            return None
        try:
            ds = nc.Dataset(match, 'r')
            dataset_cache[year] = ds

            # Print structure on first open for debugging
            if len(dataset_cache) == 1:
                print(f"\n   NC structure of {os.path.basename(match)}:")
                print(f"   Variables : {list(ds.variables.keys())}")
                print(f"   Dimensions: {list(ds.dimensions.keys())}")
                for vname in ds.variables:
                    v = ds.variables[vname]
                    if v.ndim == 1:
                        print(f"   {vname}: {float(v[0]):.3f} → "
                              f"{float(v[-1]):.3f}  (n={len(v)})")
                print()

            return ds
        except Exception as e:
            print(f"   ⚠️  Could not open {match}: {e}")
            return None

    # ── Single point extraction ───────────────────────────
    def extract_rain(lat, lon, target_date):
        target_date = pd.Timestamp(target_date)
        ds = load_year(target_date.year)
        if ds is None:
            return 0.0

        try:
            variables = list(ds.variables.keys())

            # Find rainfall variable (case-insensitive)
            rain_var = next(
                (v for v in variables
                 if v.upper() in ['RAINFALL', 'RAIN', 'RF', 'PRECIP', 'PR']),
                None
            )
            if rain_var is None:
                return 0.0

            # Identify dimension names (case-insensitive)
            dims     = list(ds.variables[rain_var].dimensions)
            lat_dim  = next((d for d in dims
                             if d.upper() in ['LATITUDE',  'LAT']), None)
            lon_dim  = next((d for d in dims
                             if d.upper() in ['LONGITUDE', 'LON']), None)
            time_dim = next((d for d in dims
                             if d.upper() in ['TIME', 'DATE', 'T']), None)

            if not all([lat_dim, lon_dim, time_dim]):
                return 0.0

            # Read coordinate arrays
            lats = np.array(ds.variables[lat_dim][:], dtype=float)
            lons = np.array(ds.variables[lon_dim][:], dtype=float)

            # ── TIME CONVERSION FIX ───────────────────────
            # Convert NC time values to days-since-epoch using the
            # units string directly — no nc.num2date() needed.
            # Then compare as plain integers (day offsets) to avoid
            # the numpy Timestamp subtraction bug that caused all-zeros.
            time_var    = ds.variables[time_dim]
            epoch       = pd.Timestamp(
                time_var.units.replace('days since ', '').strip()[:10]
            )
            # time_days: array of floats (days since epoch)
            time_days   = np.array(time_var[:], dtype=float)

            # Convert target_date to days since same epoch
            target_days = float((target_date.normalize() - epoch).days)

            # Nearest-neighbour index for each dimension
            lat_idx  = int(np.argmin(np.abs(lats - lat)))
            lon_idx  = int(np.argmin(np.abs(lons - lon)))
            time_idx = int(np.argmin(np.abs(time_days - target_days)))

            # Slice in correct dimension order (TIME, LATITUDE, LONGITUDE)
            dim_to_idx = {
                lat_dim:  lat_idx,
                lon_dim:  lon_idx,
                time_dim: time_idx
            }
            idx = tuple(dim_to_idx[d] for d in dims)
            val = float(ds.variables[rain_var][idx])

            # Guard against fill / masked values
            if np.ma.is_masked(val) or val > 1e10 or val < 0:
                return 0.0

            return val

        except Exception as e:
            return 0.0

    # ── Apply to all rows ─────────────────────────────────
    df['event_date'] = pd.to_datetime(df['event_date'])

    outside = ((df['event_date'].dt.year < RAINFALL_YEAR_MIN) |
               (df['event_date'].dt.year > RAINFALL_YEAR_MAX))
    n_outside = outside.sum()
    if n_outside > 0:
        print(f"   ⚠️  {n_outside} rows have dates outside "
              f"{RAINFALL_YEAR_MIN}–{RAINFALL_YEAR_MAX} → rainfall = 0 for these.")

    print("   Extracting event-day rainfall...")
    df['rainfall'] = df.apply(
        lambda r: extract_rain(
            r['latitude'], r['longitude'], r['event_date']),
        axis=1
    )

    print("   Extracting 3-day antecedent cumulative rainfall...")
    df['rainfall_3d_cum'] = (
        df.apply(lambda r: extract_rain(
            r['latitude'], r['longitude'],
            r['event_date'] - pd.Timedelta(days=1)), axis=1) +
        df.apply(lambda r: extract_rain(
            r['latitude'], r['longitude'],
            r['event_date'] - pd.Timedelta(days=2)), axis=1) +
        df.apply(lambda r: extract_rain(
            r['latitude'], r['longitude'],
            r['event_date'] - pd.Timedelta(days=3)), axis=1)
    )

    # Close all datasets
    for ds in dataset_cache.values():
        ds.close()

    # ── Sanity check ──────────────────────────────────────
    n_zero    = (df['rainfall'] == 0.0).sum()
    n_nonzero = (df['rainfall'] > 0.0).sum()
    print(f"\n✅ Rainfall extraction complete.")
    print(f"   Mean event-day rainfall      : {df['rainfall'].mean():.2f} mm")
    print(f"   Mean 3-day cumulative        : {df['rainfall_3d_cum'].mean():.2f} mm")
    print(f"   Non-zero rainfall rows       : {n_nonzero} / {len(df)}")
    print(f"   Zero rainfall rows           : {n_zero} / {len(df)}")

    if n_zero == len(df):
        print("\n   ❌ ALL rainfall values are 0. Possible causes:")
        print("      1. Event dates in your CSV don't fall in 2008–2017")
        print("      2. Uttarakhand lat/lon is outside NC file bounds")
        print("      3. Variable or dimension names not matched above")

    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    if not os.path.exists(INPUT_FILE):
        print("❌ Run 'preprocess.py' first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"📂 Loaded {len(df)} rows from step 1.")

    df['event_date'] = pd.to_datetime(df['event_date'])
    in_range = df[(df['event_date'].dt.year >= RAINFALL_YEAR_MIN) &
                  (df['event_date'].dt.year <= RAINFALL_YEAR_MAX)]
    print(f"   Events with rainfall coverage "
          f"({RAINFALL_YEAR_MIN}–{RAINFALL_YEAR_MAX}): "
          f"{len(in_range)} / {len(df)}")

    init_gee()
    df = get_terrain_features(df)
    df = get_rainfall_features(df)

    feature_cols = ['rainfall', 'rainfall_3d_cum',
                    'elevation', 'slope', 'aspect',
                    'twi', 'curvature', 'tpi', 'ndvi']

    before   = len(df)
    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)
    dropped  = before - len(df_clean)
    print(f"\n🗑️  Dropped {dropped} rows with NaN features "
          f"({dropped/before*100:.1f}%)")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Final dataset saved: {OUTPUT_FILE}  ({len(df_clean)} rows)")
    print(f"   Columns: {list(df_clean.columns)}")
    print(f"   Label distribution:\n"
          f"{df_clean['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()