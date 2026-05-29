---

# Uttarakhand Landslide Susceptibility Prediction & Micro-Zonation

An end-to-end Machine Learning pipeline designed to predict, evaluate, and map landslide susceptibility in Uttarakhand, India. By fusing historical event logs from the Global Landslide Catalog with high-resolution topographic data from Google Earth Engine (GEE) and gridded daily rainfall datasets, this framework provides robust macro- and micro-zonation maps under varying climatic and disaster scenarios.

---

## 📌 Project Architecture & Workflow

The framework operates via a robust three-stage pipeline designed for efficiency, high data signal, and spatial integrity:

```
  Data Preprocessing          Feature Extraction              Model Training & Mapping
+--------------------+      +--------------------+      +---------------------------------+
| Confirmed Events   |      | Google Earth Engine|      | Stratified 5-Fold CV            |
| (Positive Catalogs)|      | (SRTM DEM, Landsat)|      | Balanced Random Forest Classifier|
+---------+----------+      +---------+----------+      +----------------+----------------+
          |                           |                                  |
          v                           v                                  v
+--------------------+      +--------------------+      +---------------------------------+
| Spatial Buffer     | ---> | Gridded NetCDF     | ---> | Scenario-Based Micro-Zonation    |
| Negative Sampling  |      | Rainfall Data (.nc)|      | Maps with 3D Hillshading Overlay|
+--------------------+      +--------------------+      +---------------------------------+

```

1. **Preprocessing & Buffer-Based Negative Sampling (`preprocess.py`)**: Real landslide coordinate locations are extracted for Uttarakhand. To prevent the model from failing to distinguish between geologically identical locations, true non-landslide (negative) coordinates are randomly sampled from across the Uttarakhand bounding box using a strict $\sim$5 km (`BUFFER_DEG = 0.05`) geographic exclusion zone around known event centroids.
2. **Batch Feature Extraction (`features.py`)**: Combines remote sensing and meteorological feature engineering:
* **Terrain Features**: Pulls Elevation, Slope, Aspect, Topographic Wetness Index (TWI), Plan Curvature, and Topographic Position Index (TPI) from USGS SRTM datasets. Vegetation intensity (NDVI) is extracted via Landsat 8 composites. All geometries are batched into a **single GEE API call** (`sampleRegions`) to respect API rate-limits and maximize performance.
* **Meteorological Features**: Leverages pure binary NetCDF4 array handling to fetch exact event-day rainfall and historical 3-day cumulative antecedent rainfall without floating-point timeline bugs.


3. **Training, Explainable AI, and Visualization (`train.py`)**: Trains a Balanced Random Forest Classifier. It maps macro feature importance metrics using SHAP, queries FAO GAUL level-2 district boundaries, constructs 3D hillshaded surface textures, and projects risk micro-zonation color meshes over the terrain.

---

## 📊 Engineered Features

The dataset builds upon a robust collection of meteorological and geospatial static/dynamic feature arrays:

| Feature Name | Description | Source / Methodology |
| --- | --- | --- |
| `rainfall` | Total precipitation depth (mm) on the day of the event. | IMD RF25 Gridded NetCDF File |
| `rainfall_3d_cum` | Cumulative antecedent rainfall depth (mm) across 3 days prior. | Rolling array summation over daily NC layers |
| `elevation` | Height measured relative to Mean Sea Level (meters). | USGS SRTM GL1 (30m Resolution) |
| `slope` | Localized angle of inclination in degrees. | Extracted via native `ee.Terrain` algorithms |
| `aspect` | Compass direction of the downhill slope face. | Extracted via native `ee.Terrain` algorithms |
| `twi` | Topographic Wetness Index ($\ln(\alpha / \tan \beta)$). | Derived from HydroSHEDS flow accumulation bands |
| `curvature` | Planiform curvature (identifying convergent hollows vs convex ridges). | Computed utilizing an 8-neighbor Laplacian kernel |
| `tpi` | Topographic Position Index (evaluating ridges vs valley depths). | Elevation delta vs a 300m localized circle matrix |
| `ndvi` | Normalized Difference Vegetation Index (vegetation density scale). | Landsat 8 Level 2 Median Cloud-Masked Composites |
| `rain_slope_danger` | Multiplicative interaction term (`rainfall_3d_cum` $\times$ `slope`). | Engineered structural danger variable |

---

## ⚙️ Project Environment Setup

Follow these setup steps to configure directories and initialize dependencies:

### 1. Prerequisites & Environment

Ensure you have Python 3.8+ configured. Install the underlying data science, machine learning, and cloud-spatial dependencies:

```bash
pip install pandas numpy netCDF4 xarray scikit-learn matplotlib seaborn shap earthengine-api scipy requests

```

### 2. Directory Matrix Initialization

Execute the setup script to instantly structure the underlying repository footprint:

```bash
python setup_project.py

```

*Make sure to drop your global historical landslide timeline spreadsheet file into the newly generated `data/` folder as `Global_Landslide_Catalog_Export_rows.csv` before moving forward.*

---

## 🚀 Execution Guide

Run the pipeline sequentially to preprocess data, extract features, and train the model:

### Step 1: Preprocessing & Negative Sampling

Generates clean structural vectors and inserts spatially insulated safe points inside Uttarakhand bounds:

```bash
python preprocess.py

```

### Step 2: Cloud Spatial & Rainfall Fusion

Authenticates your Google Earth Engine account session, executes optimized vector region extractions, parses historical `.nc` matrices, drops localized `NaN` misses, and updates your training tables:

```bash
python features.py

```

### Step 3: Predictive Modeling & Advanced Visualization

Runs 5-fold cross-validation modeling, prints accuracy evaluation diagnostics, evaluates feature trees via SHAP, and creates map renders complete with 3D terrain shading:

```bash
python train.py

```

---

## 📈 Scenario Analysis & Model Outputs

Once fully executed, your pipeline automatically tests and visualizes predictions under three extreme atmospheric weather scenarios across Uttarakhand:

1. **General Monsoon Conditions**: Baseline risk evaluations using general conditions ($150\text{ mm/day}$ peak cloudburst event matching a massive $400\text{ mm}$ multi-day saturation threshold).
2. **Kedarnath Flash Flood (Historical Recall)**: Simulates the catastrophic storm event of June 17, 2013 ($81.56\text{ mm/day}$ on an existing $80.32\text{ mm}$ 3-day foundational accumulation).
3. **Dehradun Floods (Recent Validation)**: Simulates the cloudburst and flood crisis observed on September 16, 2025 ($109.69\text{ mm/day}$ paired with a massive $173.19\text{ mm}$ multi-day pre-saturation window).

### Saved Artifacts (`outputs/figures/`)

* 📊 **`confusion_matrix.png`**: Breakdown of True/False classifications across Safe/Landslide states.
* 🌲 **`feature_importance.png`**: Gini impurity metrics assessing tree splitting choices across physical indices.
* 🧠 **`shap_summary.png`**: Advanced model interpretability summary charting individual localized feature push-pull dynamics.
* 🗺️ **`susceptibility_map_*.png`**: High-resolution, production-grade micro-zonation maps. These include a high-alpine filter (`elevation > 5000m`) and a flatlands mask (`slope < 5°`) to ensure environmental accuracy, overlaid on 3D hillshaded topography and official FAO GAUL district lines.
