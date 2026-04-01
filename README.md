# Landslide Susceptibility Prediction in Uttarakhand Himalayas
### A Hybrid Machine Learning & Geospatial Approach for Disaster Management

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Earth Engine](https://img.shields.io/badge/Google_Earth_Engine-API-green)](https://earthengine.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This project implements a research-grade machine learning pipeline to predict landslide susceptibility in the Uttarakhand region of India. By fusing NASA's Global Landslide Catalog with high-resolution topographic data from **Google Earth Engine (GEE)** and gridded rainfall data from **IMD**, the system identifies the primary environmental drivers of slope failure.

The core of this project is **Explainable AI (XAI)**, using **SHAP values** to move beyond "black-box" predictions and provide geographically specific insights into why a landslide was predicted.



## 🚀 Key Features
* **Automated Geospatial Pipeline:** Seamlessly integrates local `.nc` (NetCDF) rainfall files with cloud-based terrain extraction.
* **Balanced Class Modeling:** Implements pseudo-negative sampling to address the lack of non-event data in historical catalogs, ensuring the model learns to distinguish "Safe" vs. "High Risk" zones.
* **Topographic Analysis:** Extracts 30m resolution Elevation, Slope, and Aspect via the SRTM Digital Elevation Model.
* **Model Interpretability:** Uses **SHAP (SHapley Additive exPlanations)** to visualize feature importance and dependence, providing transparency for disaster response teams.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **AI Suite:** Google Earth Engine (GEE), Scikit-Learn (Random Forest)
* **Data Science:** Pandas, NumPy, Xarray (for NetCDF processing)
* **Explainable AI:** SHAP (TreeExplainer)
* **Visualization:** Matplotlib, Seaborn

## 📂 Project Structure
```text
uk_ml_major/
├── data/               # Raw NASA CSV and IMD .nc rainfall files
├── outputs/
│   ├── figures/        # Generated SHAP and Evaluation plots (PNGs)
│   └── models/         # Cleaned and processed intermediate datasets
├── 1_preprocess.py      # Data cleaning and pseudo-negative sampling
├── 2_features.py        # GEE and Rainfall feature extraction engine
├── 3_train.py           # ML Model training, evaluation, and XAI analysis
├── setup.py            # Environment and directory verification script
└── requirements.txt     # Project dependencies

```

## ⚙️ Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/Akd2510/uk_ml_major.git](https://github.com/Akd2510/uk_ml_major.git)
cd uk_ml_major

```


2. **Initialize the environment:**
```bash
python setup.py

```


3. **Authenticate Google Earth Engine:**
Ensure you have a [Google Cloud Project](https://code.earthengine.google.com/) with Earth Engine API enabled. Update the `MY_PROJECT_ID` in `2_features.py`.

## 📈 Methodology & Results

The pipeline follows a structured research workflow:

1. **Pre-processing:** Filters historical events for Uttarakhand and generates synthetic negative samples to balance the dataset.
2. **Feature Fusion:** Joins satellite terrain data (SRTM) with temporal rainfall values from IMD gridded datasets.
3. **Modeling & XAI:** Trains a Random Forest Classifier and evaluates it using AUC-ROC. SHAP analysis is then performed to identify dominant triggers.

### Performance

The model consistently achieves an **AUC-ROC score of >0.85**. Our analysis identifies **Daily Rainfall** and **Slope Angle** as the dominant factors, with landslide risk significantly increasing on slopes greater than 25° during high-intensity precipitation events.

## 🤝 Contributing

This project is an evolution of the [original repository](https://github.com/abhimanyuxingh/uk_ml_major). Contributions regarding real-time sensor integration or Deep Learning (CNN-LSTM) extensions are welcome.

## 📄 License

This project is licensed under the MIT License.

```

```
