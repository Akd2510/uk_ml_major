import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# CONFIG
INPUT_FILE = os.path.join('outputs', 'models', 'step2_final_dataset.csv')
FIG_DIR = os.path.join('outputs', 'figures')

def train_model():
    print("--- Step 3: Training & Evaluation ---")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ Run '2_features.py' first.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Define Features (X) and Target (y)
    features = ['rainfall', 'elevation', 'slope', 'aspect']
    X = df[features]
    y = df['label']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("🚀 Model Trained Successfully.")

    # --- RESULTS ---
    print("\nClassification Report:")
    print(classification_report(y_test, model.predict(X_test)))
    
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"⭐️ AUC-ROC Score: {auc:.2f}")

    # --- PLOTS FOR PAPER ---
    
    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Safe', 'Landslide'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix (AUC = {auc:.2f})')
    plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'))
    plt.close()

    # 2. SHAP Explanation
    print("🧠 Generating SHAP plots...")
    
    # Create the explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Check if shap_values is a list (Old SHAP) or an array (New SHAP)
    if isinstance(shap_values, list):
        # It's a list of [Class 0, Class 1]. We want Class 1 (Landslides)
        shap_data = shap_values[1]
    else:
        # It's already a single matrix for the positive class
        shap_data = shap_values

    # Check dimensions to be safe
    if shap_data.shape != X_test.shape:
        # Fallback for 3D arrays (samples, features, classes)
        if len(shap_data.shape) == 3:
            shap_data = shap_data[:, :, 1]

    plt.figure()
    shap.summary_plot(shap_data, X_test, show=False)
    plt.title('Feature Importance (SHAP)')
    plt.savefig(os.path.join(FIG_DIR, 'shap_summary.png'), bbox_inches='tight')
    plt.close()

    print(f"✅ All plots saved to: {FIG_DIR}")

if __name__ == "__main__":
    train_model()