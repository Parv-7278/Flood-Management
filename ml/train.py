"""
FloodWatch-Himalaya: Machine Learning Training Pipeline.
Trains Random Forest flood inundation classifier and Gradient Boosting road risk regressor
with geographic train/test separation across Himalayan river sectors.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    jaccard_score, brier_score_loss, mean_absolute_error, r2_score
)

from ml.config import (
    FEATURE_NAMES,
    ROAD_FEATURE_NAMES,
    FLOOD_CLASSIFIER_PATH,
    ROAD_REGRESSOR_PATH,
    METADATA_PATH,
    EVALUATION_REPORT_PATH,
    RF_PARAMS,
    GB_PARAMS,
    MODELS_DIR
)
from ml.features import FeaturePipeline


def train_flood_models() -> Dict[str, Any]:
    """
    Executes end-to-end model training:
    1. Generates geophysically grounded dataset across 4 sectors
    2. Performs geographic out-of-basin split (Train: Upper/Middle/Trishuli, Test: Lower Barhabise)
    3. Trains RandomForestClassifier and GradientBoostingRegressor
    4. Computes rigorous test metrics (IoU, F1, Precision, Recall, ROC-AUC, MAE, R²)
    5. Saves serialized model artifacts and metadata
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("🛰️  FLOODWATCH-HIMALAYA: AI/ML MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Generate Datasets
    print("\n[1/4] Generating geophysically grounded feature dataset...")
    df = FeaturePipeline.generate_training_dataset(num_samples_per_sector=1500, random_seed=42)
    print(f"  ✓ Generated {len(df):,} grid cell observations across 4 Himalayan sectors")

    # 2. Geographic Train/Test Split (Spatial cross-validation to prevent spatial autocorrelation)
    # Training set: Upper Bhote Koshi, Middle Bhote Koshi, Trishuli
    # Test set: Lower Bhote Koshi (Barhabise sector - held out completely)
    train_mask = df["sector_id"].isin(["bhote_upper", "bhote_middle", "trishuli_west"])
    test_mask = df["sector_id"] == "bhote_lower"

    X_train = df.loc[train_mask, FEATURE_NAMES]
    y_train = df.loc[train_mask, "is_inundated"]
    X_test = df.loc[test_mask, FEATURE_NAMES]
    y_test = df.loc[test_mask, "is_inundated"]

    print(f"  ✓ Geographic Separation Split:")
    print(f"    - Training set: {len(X_train):,} samples (Sectors: Upper/Middle Bhote Koshi & Trishuli)")
    print(f"    - Test set:     {len(X_test):,} samples (Sector: Lower Barhabise - Out-of-basin validation)")

    # 3. Train Flood Inundation Random Forest
    print("\n[2/4] Training Random Forest Flood Inundation Classifier...")
    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(X_train, y_train)

    # Predictions on unseen geographic sector
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    # Metrics
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    iou_jaccard = float(jaccard_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))

    print("  ✓ Spatial Inundation Metrics (Unseen Test Sector):")
    print(f"    - Precision:                  {precision * 100:.2f}%")
    print(f"    - Recall (Sensitivity):        {recall * 100:.2f}%")
    print(f"    - F1-Score:                   {f1 * 100:.2f}%")
    print(f"    - Intersection over Union:    {iou_jaccard * 100:.2f}% (IoU / Jaccard)")
    print(f"    - ROC-AUC:                    {roc_auc:.4f}")
    print(f"    - Brier Calibration Score:    {brier:.4f}")

    # Feature Importances
    importances = dict(zip(FEATURE_NAMES, [round(float(v), 4) for v in rf_model.feature_importances_]))
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    print("\n  ✓ Top 5 Contributing Risk Features:")
    for feat, imp in list(sorted_importances.items())[:5]:
        print(f"    • {feat:25s}: {imp * 100:.2f}%")

    # 4. Train Road Inundation Gradient Boosting Regressor
    print("\n[3/4] Training Gradient Boosting Road Inundation Regressor...")
    road_df = FeaturePipeline.generate_road_training_dataset(num_samples=1000)
    X_road = road_df[ROAD_FEATURE_NAMES]
    y_road = road_df["flooded_length_m"]

    # 80/20 train/test split
    split_idx = int(len(road_df) * 0.8)
    X_r_train, X_r_test = X_road.iloc[:split_idx], X_road.iloc[split_idx:]
    y_r_train, y_r_test = y_road.iloc[:split_idx], y_road.iloc[split_idx:]

    gb_model = GradientBoostingRegressor(**GB_PARAMS)
    gb_model.fit(X_r_train, y_r_train)
    r_pred = gb_model.predict(X_r_test)

    road_mae = float(mean_absolute_error(y_r_test, r_pred))
    road_r2 = float(r2_score(y_r_test, r_pred))
    print(f"  ✓ Road Risk Model MAE: {road_mae:.1f} meters | R² Score: {road_r2:.3f}")

    # 5. Serialize Artifacts & Metadata
    print("\n[4/4] Serializing model binaries and evaluation report...")
    joblib.dump(rf_model, FLOOD_CLASSIFIER_PATH)
    joblib.dump(gb_model, ROAD_REGRESSOR_PATH)

    metadata = {
        "model_version": "v2.6-production",
        "training_date": datetime.utcnow().isoformat() + "Z",
        "model_type": "RandomForestClassifier (Ensemble Decision Trees)",
        "framework": "scikit-learn 1.9.1",
        "feature_names": FEATURE_NAMES,
        "feature_importances": sorted_importances,
        "metrics": {
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_iou_jaccard": iou_jaccard,
            "test_roc_auc": roc_auc,
            "test_brier_score": brier,
            "road_mae_meters": road_mae,
            "road_r2_score": road_r2
        },
        "geographic_split": {
            "train_sectors": ["bhote_upper", "bhote_middle", "trishuli_west"],
            "test_sector": "bhote_lower (Barhabise confluence)"
        },
        "status": "TRAINED & VALIDATED"
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved models to {MODELS_DIR}/")
    print(f"  ✓ Metadata saved to {METADATA_PATH}")
    print("=" * 70)
    print("✨ Model training completed successfully!")
    return metadata


if __name__ == "__main__":
    train_flood_models()
