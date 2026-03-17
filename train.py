import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import shap
import ee
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay,
                              RocCurveDisplay)
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_FILE    = os.path.join('outputs', 'models', 'step2_final_dataset.csv')
FIG_DIR       = os.path.join('outputs', 'figures')
MY_PROJECT_ID = 'landslide-ml-2026'   # <── change if needed

# Features fed to the model  (added twi, curvature, tpi, ndvi, rainfall_3d_cum)
FEATURES = ['rainfall', 'rainfall_3d_cum',
            'elevation', 'slope', 'aspect',
            'twi', 'curvature', 'tpi', 'ndvi']

# Susceptibility map grid settings
MAP_LAT_MIN, MAP_LAT_MAX = 28.7, 31.5
MAP_LON_MIN, MAP_LON_MAX = 77.5, 81.1
MAP_RESOLUTION           = 0.15   # degrees (~15 km) — increase to 0.02 for finer map
                                   # but GEE will take longer

# Rainfall to simulate for the map (extreme monsoon scenario)
MAP_RAINFALL_DAY    = 150.0   # mm/day
MAP_RAINFALL_3D_CUM = 400.0   # mm cumulative 3 days


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
def train_model():
    print("--- Step 3: Training & Evaluation ---")

    if not os.path.exists(INPUT_FILE):
        print("❌ Run 'features.py' first.")
        return

    df = pd.read_csv(INPUT_FILE)
    os.makedirs(FIG_DIR, exist_ok=True)

    # Use only features that are actually in the file
    # (graceful fallback if old dataset without new features is used)
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"⚠️  Missing features (run features.py again to get them): {missing}")
    print(f"✅ Using features: {available}")

    X = df[available].values
    y = df['label'].values

    # ── Random Forest ─────────────────────────────────────
    #
    # KEY CHANGES from original:
    #   class_weight='balanced'  → prevents model ignoring minority class
    #   n_estimators=300         → more stable predictions, less variance
    #   min_samples_leaf=5       → prevents overfitting tiny leaf nodes
    #   oob_score=True           → free internal validation estimate
    #
    model = RandomForestClassifier(
        n_estimators    = 300,
        class_weight    = 'balanced',
        min_samples_leaf= 5,
        random_state    = 42,
        n_jobs          = -1,
        oob_score       = True,
    )

    # ── Stratified K-Fold CV ──────────────────────────────
    #
    # WHY not simple train_test_split:
    #   Spatial autocorrelation means nearby pixels share features.
    #   A random 80/20 split leaks information → inflated accuracy.
    #   Stratified K-Fold gives more honest performance estimates.
    #
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    print("\n📊 5-Fold Cross-Validation:")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model.fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[val_idx])[:, 1]
        auc  = roc_auc_score(y[val_idx], prob)
        auc_scores.append(auc)
        print(f"  Fold {fold+1}: AUC-ROC = {auc:.4f}")

    print(f"\n⭐ Mean AUC-ROC: {np.mean(auc_scores):.4f} "
          f"± {np.std(auc_scores):.4f}")

    # ── Final model on full data ───────────────────────────
    model.fit(X, y)
    print(f"   OOB Score:   {model.oob_score_:.4f}  "
          f"(out-of-bag — another honest accuracy estimate)")

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print("\nClassification Report (full training data):")
    print(classification_report(y, y_pred, target_names=['Safe', 'Landslide']))

    # ── Feature importance ────────────────────────────────
    importance_df = pd.DataFrame({
        'feature':    available,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n🌲 Feature Importances (Gini):")
    print(importance_df.to_string(index=False))

    # ─────────────────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────────────────

    # 1. Confusion Matrix
    cm   = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Safe', 'Landslide'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix  (OOB AUC ≈ {model.oob_score_:.2f})')
    fig.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. Feature Importance Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ['#d73027' if i < 3 else '#4575b4' for i in range(len(importance_df))]
    ax.barh(importance_df['feature'][::-1],
            importance_df['importance'][::-1],
            color=colors[::-1])
    ax.set_xlabel('Gini Importance')
    ax.set_title('Feature Importance')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3. SHAP Summary
    print("\n🧠 Generating SHAP explanation...")
    X_df     = pd.DataFrame(X, columns=available)
    sample_n = min(300, len(X_df))
    X_sample = X_df.sample(sample_n, random_state=42)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Handle both old (list) and new (array) SHAP API
    if isinstance(shap_values, list):
        shap_data = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_data = shap_values[:, :, 1]
    else:
        shap_data = shap_values

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_data, X_sample, show=False)
    plt.title('SHAP Feature Importance')
    fig.savefig(os.path.join(FIG_DIR, 'shap_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.close('all')

    print(f"✅ Plots saved to: {FIG_DIR}")
    return model, available


# ─────────────────────────────────────────────
# SUSCEPTIBILITY MAP GENERATION
# ─────────────────────────────────────────────
def generate_susceptibility_map(model, feature_names):
    print("\n🗺️  Generating Susceptibility Map...")

    try:
        ee.Initialize(project=MY_PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=MY_PROJECT_ID)

    # ── Build feature image on GEE side ──────────────────
    srtm      = ee.Image('USGS/SRTMGL1_003')
    terrain   = ee.Terrain.products(srtm)
    elev      = srtm.rename('elevation')

    slope_rad  = terrain.select('slope').multiply(np.pi / 180)
    tan_slope  = slope_rad.tan().max(ee.Image(0.001))
    flow_accum = ee.Image('WWF/HydroSHEDS/15ACC').select('b1').rename('flow_accum')
    twi        = flow_accum.divide(tan_slope).log().rename('twi')
    curvature  = elev.convolve(ee.Kernel.laplacian8(normalize=False)).rename('curvature')
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

    image = (terrain.select(['elevation', 'slope', 'aspect'])
             .addBands(twi).addBands(curvature)
             .addBands(tpi).addBands(ndvi))

    # ── Sample the entire region in ONE batch call ────────
    region = ee.Geometry.Rectangle(
        [MAP_LON_MIN, MAP_LAT_MIN, MAP_LON_MAX, MAP_LAT_MAX])

    print("   Sampling grid from GEE in one batch (this takes ~1 min)...")
    sample = image.sample(
        region=region,
        scale=5000,          # 5 km resolution — one call, no loop needed
        geometries=True,
        seed=42
    )

    features_info = sample.getInfo()
    print(f"   Got {len(features_info['features'])} grid points from GEE.")

    # ── Parse into dataframe ──────────────────────────────
    records = []
    for f in features_info['features']:
        p   = f['properties']
        geo = f['geometry']['coordinates']
        records.append({
            'longitude': geo[0],
            'latitude':  geo[1],
            'elevation': p.get('elevation', np.nan),
            'slope':     p.get('slope',     np.nan),
            'aspect':    p.get('aspect',    np.nan),
            'twi':       p.get('twi',       np.nan),
            'curvature': p.get('curvature', np.nan),
            'tpi':       p.get('tpi',       np.nan),
            'ndvi':      p.get('ndvi',      np.nan),
        })

    grid_df = pd.DataFrame(records)
    grid_df['rainfall']        = MAP_RAINFALL_DAY
    grid_df['rainfall_3d_cum'] = MAP_RAINFALL_3D_CUM
    grid_df = grid_df.fillna(grid_df.median(numeric_only=True))

    # ── Predict ───────────────────────────────────────────
    X_grid   = grid_df[feature_names].values
    probs    = model.predict_proba(X_grid)[:, 1]
    grid_df['probability'] = probs

    # ── Smooth & mask ─────────────────────────────────────
    # Pivot to 2D grid for smoothing
    grid_df = grid_df.sort_values(['latitude', 'longitude'])
    lats_u  = np.sort(grid_df['latitude'].unique())
    lons_u  = np.sort(grid_df['longitude'].unique())
    prob_2d = grid_df.pivot_table(
        index='latitude', columns='longitude',
        values='probability', aggfunc='mean'
    ).values

    prob_smooth = gaussian_filter(prob_2d, sigma=1.5)

    # Mask flat terrain
    slope_2d = grid_df.pivot_table(
        index='latitude', columns='longitude',
        values='slope', aggfunc='mean'
    ).values
    prob_smooth[slope_2d < 5] = 0.0
    prob_smooth = np.clip(prob_smooth, 0, 1)

    # ── Plot ──────────────────────────────────────────────
    lon_grid, lat_grid = np.meshgrid(lons_u, lats_u)

    bounds     = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels_cls = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    colors_cls = ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c']
    cmap       = mcolors.ListedColormap(colors_cls)
    norm       = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.pcolormesh(lon_grid, lat_grid, prob_smooth,
                  cmap=cmap, norm=norm, shading='auto')

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, ticks=bounds, shrink=0.7)
    cbar.set_label('Susceptibility Probability', fontsize=11)
    for i, label in enumerate(labels_cls):
        cbar.ax.text(1.5, (bounds[i] + bounds[i+1]) / 2,
                     label, va='center', fontsize=9)

    ax.set_title(
        'Landslide Susceptibility Map — Uttarakhand\n'
        f'(Simulated: {MAP_RAINFALL_DAY} mm/day, '
        f'{MAP_RAINFALL_3D_CUM} mm 3-day cumulative)',
        fontsize=12, pad=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    out_path = os.path.join(FIG_DIR, 'susceptibility_map.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Susceptibility map saved → {out_path}")

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    result = train_model()
    if result is not None:
        trained_model, feat_names = result
        generate_susceptibility_map(trained_model, feat_names)