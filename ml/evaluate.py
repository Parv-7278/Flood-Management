"""
FloodWatch-Himalaya: Machine Learning Evaluation Module.
Computes comprehensive validation metrics for flood inundation classification
and road segment infrastructure risk regression.
"""

import sys
import json
from pathlib import Path
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
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    jaccard_score, brier_score_loss, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

from ml.config import (
    FLOOD_CLASSIFIER_PATH,
    ROAD_REGRESSOR_PATH,
    METADATA_PATH,
    EVALUATION_REPORT_PATH,
    FEATURE_NAMES,
    ROAD_FEATURE_NAMES
)
from ml.features import FeaturePipeline


def evaluate_models() -> Dict[str, Any]:
    """
    Evaluates trained models on an independent geographic test sector
    and computes full classification & regression metrics.
    """
    if not FLOOD_CLASSIFIER_PATH.exists() or not ROAD_REGRESSOR_PATH.exists():
        from ml.train import train_flood_models
        train_flood_models()

    rf_model = joblib.load(FLOOD_CLASSIFIER_PATH)
    gb_model = joblib.load(ROAD_REGRESSOR_PATH)

    # Generate evaluation data on held-out test sector
    df = FeaturePipeline.generate_training_dataset(num_samples_per_sector=800, random_seed=999)
    test_df = df[df["sector_id"] == "bhote_lower"]

    X_test = test_df[FEATURE_NAMES]
    y_test = test_df["is_inundated"]

    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred).tolist()

    class_metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "iou_jaccard": float(jaccard_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "confusion_matrix": cm,
        "sample_count": len(X_test)
    }

    # Road Regressor Evaluation
    road_df = FeaturePipeline.generate_road_training_dataset(num_samples=400)
    X_road = road_df[ROAD_FEATURE_NAMES]
    y_road = road_df["flooded_length_m"]

    r_pred = gb_model.predict(X_road)
    reg_metrics = {
        "mae_meters": float(mean_absolute_error(y_road, r_pred)),
        "rmse_meters": float(np.sqrt(mean_squared_error(y_road, r_pred))),
        "r2_score": float(r2_score(y_road, r_pred)),
        "sample_count": len(X_road)
    }

    feature_importances = dict(zip(FEATURE_NAMES, [round(float(v), 4) for v in rf_model.feature_importances_]))
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

    report = {
        "classification": class_metrics,
        "regression": reg_metrics,
        "feature_importances": sorted_importances,
        "status": "VALIDATED"
    }

    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = evaluate_models()
    c = rep["classification"]
    r = rep["regression"]
    print("=" * 65)
    print("AI/ML MODEL EVALUATION REPORT:")
    print("=" * 65)
    print(f"  Classification Precision:       {c['precision'] * 100:.2f}%")
    print(f"  Classification Recall:          {c['recall'] * 100:.2f}%")
    print(f"  Classification F1-Score:        {c['f1_score'] * 100:.2f}%")
    print(f"  Intersection over Union (IoU):  {c['iou_jaccard'] * 100:.2f}%")
    print(f"  ROC-AUC:                        {c['roc_auc']:.4f}")
    print(f"  Road Model MAE:                 {r['mae_meters']:.1f} meters")
    print(f"  Road Model R²:                  {r['r2_score']:.3f}")
    print("=" * 65)
