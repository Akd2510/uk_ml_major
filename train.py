import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource
import seaborn as sns
import shap
import ee
import os
import requests
import zipfile
import io
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay)
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_FILE    = os.path.join('outputs', 'models', 'step2_final_dataset.csv')
FIG_DIR       = os.path.join('outputs', 'figures')
TERRAIN_CACHE = os.path.join('outputs', 'models', 'map_terrain_grid.csv')
MY_PROJECT_ID = 'landslide-ml-2026'

FEATURES = ['rainfall', 'rainfall_3d_cum',
            'elevation', 'slope', 'aspect',
            'twi', 'curvature', 'tpi', 'ndvi']

# Map grid
MAP_LAT_MIN, MAP_LAT_MAX = 28.7, 31.5
MAP_LON_MIN, MAP_LON_MAX = 77.5, 81.1
MAP_RESOLUTION           = 0.02   # ~8 km

# ── 3 Scenarios ───────────────────────────────
# Rainfall values from actual NC files or realistic simulation
SCENARIOS = {
    'General Monsoon': {
        'rainfall':        150.0,
        'rainfall_3d_cum': 400.0,
        'filename':        'susceptibility_map_monsoon.png',
        'title':           'General Monsoon Scenario\n(150 mm/day, 400 mm 3-day cumulative)',
    },
    'Kedarnath 2013': {
        'rainfall':        81.56,   # actual Jun 17 2013 from NC file
        'rainfall_3d_cum': 80.32,   # Jun 14+15+16 = 11.55+22.74+46.03
        'filename':        'susceptibility_map_kedarnath2013.png',
        'title':           'Kedarnath Flash Flood — June 17, 2013\n(81.6 mm/day, 80.3 mm 3-day cumulative)',
    },
    'Dehradun 2025': {
        'rainfall':        109.69,  # actual Sep 16 2025 from NC file
        'rainfall_3d_cum': 173.19,  # Sep 13+14+15 = 116.52+35.74+20.93
        'filename':        'susceptibility_map_dehradun2025.png',
        'title':           'Dehradun Floods — September 16, 2025\n(109.7 mm/day, 173.2 mm 3-day cumulative)',
    },
}


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
def train_model():
    print("--- Step 3: Training & Evaluation ---")

    if not os.path.exists(INPUT_FILE):
        print("❌ Run 'features.py' first.")
        return None, None

    df = pd.read_csv(INPUT_FILE)
    os.makedirs(FIG_DIR, exist_ok=True)

    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"⚠️  Missing features: {missing} — run features.py again")
    print(f"✅ Using features: {available}")

    X = df[available].values
    y = df['label'].values

    model = RandomForestClassifier(
        n_estimators     = 300,
        class_weight     = 'balanced',
        min_samples_leaf = 5,
        random_state     = 42,
        n_jobs           = -1,
        oob_score        = True,
    )

    # ── 5-Fold Stratified CV ──────────────────────────────
    skf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    print("\n📊 5-Fold Cross-Validation:")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model.fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[val_idx])[:, 1]
        auc  = roc_auc_score(y[val_idx], prob)
        auc_scores.append(auc)
        print(f"  Fold {fold+1}: AUC-ROC = {auc:.4f}")

    print(f"\n⭐ Mean AUC-ROC : {np.mean(auc_scores):.4f} "
          f"± {np.std(auc_scores):.4f}")

    # Final model on full data
    model.fit(X, y)
    print(f"   OOB Score    : {model.oob_score_:.4f}")

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Safe', 'Landslide']))

    # ── Feature importance ────────────────────────────────
    importance_df = pd.DataFrame({
        'feature':    available,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n🌲 Feature Importances:")
    print(importance_df.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────

    # 1. Confusion Matrix
    cm   = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Safe', 'Landslide'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix  (OOB Score = {model.oob_score_:.2f})')
    fig.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. Feature Importance Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ['#d73027' if i < 3 else '#4575b4'
               for i in range(len(importance_df))]
    ax.barh(importance_df['feature'][::-1],
            importance_df['importance'][::-1],
            color=colors[::-1])
    ax.set_xlabel('Gini Importance')
    ax.set_title('Feature Importance')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'feature_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3. SHAP Summary
    print("\n🧠 Generating SHAP explanation...")
    X_df     = pd.DataFrame(X, columns=available)
    X_sample = X_df.sample(min(300, len(X_df)), random_state=42)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_data = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_data = shap_values[:, :, 1]
    else:
        shap_data = shap_values

    shap.summary_plot(shap_data, X_sample, show=False)
    plt.title('SHAP Feature Importance')
    plt.savefig(os.path.join(FIG_DIR, 'shap_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.close('all')

    print(f"✅ Training plots saved to: {FIG_DIR}")
    return model, available


# ─────────────────────────────────────────────
# TERRAIN GRID — cached so GEE is called once
# ─────────────────────────────────────────────
def get_or_build_terrain_grid():
    """
    Fetches terrain features for the map grid from GEE.
    Saves to CSV on first run — reuses it on every subsequent run.
    """
    if os.path.exists(TERRAIN_CACHE):
        print(f"✅ Loading cached terrain grid from {TERRAIN_CACHE}")
        return pd.read_csv(TERRAIN_CACHE)

    print("📡 Building high-res terrain grid from GEE (one-time, will be cached)...")

    try:
        ee.Initialize(project=MY_PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=MY_PROJECT_ID)

    # Build feature image
    srtm      = ee.Image('USGS/SRTMGL1_003')
    terrain   = ee.Terrain.products(srtm)
    elev      = srtm.rename('elevation')

    slope_rad  = terrain.select('slope').multiply(np.pi / 180)
    tan_slope  = slope_rad.tan().max(ee.Image(0.001))
    flow_accum = (ee.Image('WWF/HydroSHEDS/15ACC')
                  .select('b1').rename('flow_accum'))
    twi        = flow_accum.divide(tan_slope).log().rename('twi')
    curvature  = elev.convolve(
        ee.Kernel.laplacian8(normalize=False)).rename('curvature')
    mean_elev  = elev.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.circle(radius=10, units='pixels'))
    tpi        = elev.subtract(mean_elev).rename('tpi')

    ls8  = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterDate('2015-01-01', '2023-12-31')
            .filter(ee.Filter.lt('CLOUD_COVER', 20)).median())
    nir  = ls8.select('SR_B5').multiply(0.0000275).add(-0.2)
    red  = ls8.select('SR_B4').multiply(0.0000275).add(-0.2)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('ndvi')

    # Combine features AND add Pixel Coordinates (Longitude/Latitude)
    image = (terrain.select(['elevation', 'slope', 'aspect'])
             .addBands(twi).addBands(curvature)
             .addBands(tpi).addBands(ndvi)
             .addBands(ee.Image.pixelLonLat())) # <-- FIX: This automatically adds lat/lon bands

    region = ee.Geometry.Rectangle(
        [MAP_LON_MIN, MAP_LAT_MIN, MAP_LON_MAX, MAP_LAT_MAX])

    print("   Sampling grid from GEE... (this might take a moment for high-res)")
    sample  = image.sample(
        region=region,
        scale=int(MAP_RESOLUTION * 111000),  # degrees → approximate metres
        geometries=False, # <-- FIX: Set to False because coords are now bands
        seed=42
    )

    # <-- FIX: Bypass the 5000 element limit by downloading directly as CSV via URL
    try:
        url = sample.getDownloadURL(filetype='csv')
        response = requests.get(url)
        grid_df = pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        print(f"❌ Error downloading from GEE: {e}")
        return None

    print(f"   GEE returned {len(grid_df)} grid points.")

    # Fill NaNs with medians and clean up unneeded columns
    grid_df = grid_df.fillna(grid_df.median(numeric_only=True))
    if 'system:index' in grid_df.columns:
        grid_df = grid_df.drop(columns=['system:index'])
    if '.geo' in grid_df.columns:
        grid_df = grid_df.drop(columns=['.geo'])

    os.makedirs(os.path.dirname(TERRAIN_CACHE), exist_ok=True)
    grid_df.to_csv(TERRAIN_CACHE, index=False)
    print(f"✅ Terrain grid cached → {TERRAIN_CACHE}")
    return grid_df

# ─────────────────────────────────────────────
# DISTRICT BOUNDARIES
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# DISTRICT BOUNDARIES (USING GOOGLE EARTH ENGINE)
# ─────────────────────────────────────────────
def get_uttarakhand_districts():
    """
    Fetches official Uttarakhand district boundaries directly from 
    Google Earth Engine (FAO GAUL dataset) instead of a web URL.
    """
    print("🗺️  Fetching Uttarakhand district boundaries from GEE...")
    try:
        import ee
        
        # Ensure GEE is initialized
        try:
            ee.Number(1).getInfo()
        except Exception:
            ee.Initialize(project=MY_PROJECT_ID)

        # Access the FAO GAUL dataset (Level 2 = Districts)
        districts_fc = ee.FeatureCollection("FAO/GAUL/2015/level2") \
                         .filter(ee.Filter.eq('ADM1_NAME', 'Uttarakhand'))
        
        # Download the coordinates
        features = districts_fc.getInfo()['features']
        
        districts = []
        for feat in features:
            # ADM2_NAME is the district name in the GAUL dataset
            name = feat['properties'].get('ADM2_NAME', 'Unknown')
            geom = feat['geometry']
            
            # Extract polygon coordinates
            if geom['type'] == 'Polygon':
                coords = geom['coordinates'][0]
                districts.append((name, coords))
            elif geom['type'] == 'MultiPolygon':
                for poly in geom['coordinates']:
                    districts.append((name, poly[0]))
                    
        print(f"   ✅ Loaded {len(districts)} district polygons directly from Cloud.")
        return districts
        
    except Exception as e:
        print(f"   ⚠️  GEE district fetch failed ({e}) — map will still generate.")
        return None


# ─────────────────────────────────────────────
# SINGLE SCENARIO MAP
# ─────────────────────────────────────────────
def generate_scenario_map(model, feature_names, grid_df,
                          scenario_name, scenario_cfg, districts):
    """Generates a highly realistic susceptibility map with 3D hillshading."""

    print(f"\n🗺️  Generating realistic map: {scenario_name}")

    gdf = grid_df.copy()
    gdf['rainfall']        = scenario_cfg['rainfall']
    gdf['rainfall_3d_cum'] = scenario_cfg['rainfall_3d_cum']
    gdf = gdf.fillna(gdf.median(numeric_only=True))

    # Predict Probabilities
    X_grid = gdf[feature_names].values
    probs  = model.predict_proba(X_grid)[:, 1]
    gdf['probability'] = probs

    # Pivot to 2D Arrays
    gdf      = gdf.sort_values(['latitude', 'longitude'])
    lats_u   = np.sort(gdf['latitude'].unique())
    lons_u   = np.sort(gdf['longitude'].unique())
    
    prob_2d  = gdf.pivot_table(index='latitude', columns='longitude', values='probability', aggfunc='mean').values
    slope_2d = gdf.pivot_table(index='latitude', columns='longitude', values='slope', aggfunc='mean').values
    elev_2d  = gdf.pivot_table(index='latitude', columns='longitude', values='elevation', aggfunc='mean').values

    # Post-processing & Smart Masking
    prob_smooth = gaussian_filter(prob_2d, sigma=1.5) # Lowered sigma because grid is finer
    
    # Realistic Masks: No landslides on flat plains OR permanent glaciers (>5000m)
    prob_smooth[slope_2d < 5] = 0.0          
    prob_smooth[elev_2d > 5000] = np.nan # High-alpine mask
    
    prob_smooth = np.clip(prob_smooth, 0, 1)

    # ── Plotting ──────────────────────────────────────────────
    bounds     = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cls_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    cls_colors = ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c']
    cmap       = mcolors.ListedColormap(cls_colors)
    cmap.set_bad(color='white', alpha=0) # Make NaN (glaciers) transparent
    norm       = mcolors.BoundaryNorm(bounds, cmap.N)

    lon_grid, lat_grid = np.meshgrid(lons_u, lats_u)

    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')

    # 1. Generate the 3D Hillshade Background
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(elev_2d, vert_exag=1.5, dx=MAP_RESOLUTION, dy=MAP_RESOLUTION)
    ax.pcolormesh(lon_grid, lat_grid, hillshade, cmap='gray', shading='gouraud', zorder=1)

    # 2. Overlay the Susceptibility colors with transparency (alpha=0.65)
    ax.pcolormesh(lon_grid, lat_grid, prob_smooth, cmap=cmap, norm=norm, 
                  shading='auto', zorder=2, alpha=0.65)

    # ── District boundaries overlay ───────────────────────
    if districts:
        for name, coords in districts:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.plot(xs, ys, color='#222222', linewidth=1.2, zorder=3, alpha=0.8) # Thicker, darker borders
            
            cx, cy = np.mean(xs), np.mean(ys)
            if (MAP_LON_MIN < cx < MAP_LON_MAX and MAP_LAT_MIN < cy < MAP_LAT_MAX):
                ax.text(cx, cy, name, fontsize=7, ha='center', va='center',
                        color='black', zorder=4, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='gray'))

    # ── Colorbar ──────────────────────────────────────────
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, ticks=bounds, shrink=0.65, pad=0.08)
    cbar.set_label('Landslide Susceptibility Probability', fontsize=11, weight='bold')
    cbar.ax.set_yticklabels([f'{b:.1f}' for b in bounds])

    # Class label patches
    patches = [mpatches.Patch(color=cls_colors[i], label=cls_labels[i]) for i in range(5)]
    patches.append(mpatches.Patch(color='white', label='Glacier/Flat (Masked)', alpha=0.5))
    ax.legend(handles=patches, loc='lower left', fontsize=9, title='Risk Class',
              title_fontsize=10, framealpha=0.9, edgecolor='black')

    ax.set_xlim(MAP_LON_MIN, MAP_LON_MAX)
    ax.set_ylim(MAP_LAT_MIN, MAP_LAT_MAX)
    ax.set_title(f'Landslide Susceptibility Micro-Zonation\n{scenario_cfg["title"]}', 
                 fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude',  fontsize=11)
    
    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out_path = os.path.join(FIG_DIR, scenario_cfg['filename'])
    fig.savefig(out_path, dpi=300, bbox_inches='tight') # Boosted DPI to 300 for crisp text
    plt.close('all')
    print(f"   ✅ Saved Realistic Map → {out_path}")
# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Train model
    trained_model, feat_names = train_model()
    if trained_model is None:
        exit(1)

    # 2. Get terrain grid (cached after first run)
    grid_df = get_or_build_terrain_grid()

    # 3. Download district boundaries
    districts = get_uttarakhand_districts()

    # 4. Generate one map per scenario
    os.makedirs(FIG_DIR, exist_ok=True)
    for name, cfg in SCENARIOS.items():
        generate_scenario_map(
            trained_model, feat_names,
            grid_df, name, cfg, districts)

    print(f"\n✅ All maps saved to: {FIG_DIR}")