"""
FloodWatch-Himalaya: Machine Learning Configuration & Feature Definitions.
Defines model directories, feature sets, targets, and validated scenario presets.
"""

from pathlib import Path
from typing import Dict, Any, List

# Paths
ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

FLOOD_CLASSIFIER_PATH = MODELS_DIR / "flood_rf_model.joblib"
ROAD_REGRESSOR_PATH = MODELS_DIR / "road_gb_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
EVALUATION_REPORT_PATH = MODELS_DIR / "evaluation_report.json"

# Feature Definitions
FEATURE_NAMES: List[str] = [
    # Topographic / DEM-30 features
    "elevation_m",
    "slope_deg",
    "valley_dist_km",
    "flow_accumulation_proxy",
    # Hydrological & Meteorological features
    "rainfall_intensity_mmh",
    "rainfall_duration_hrs",
    "cumulative_precip_mm",
    "surge_height_m",
    "upstream_discharge_m3s",
    # Thermal & Hazard Propagation features
    "thermal_anomaly_k",
    "propagation_factor",
    # Infrastructure Exposure features
    "building_density_sqkm",
    "road_proximity_m",
    "bridge_proximity_m"
]

ROAD_FEATURE_NAMES: List[str] = [
    "length_m",
    "mean_elevation_m",
    "mean_slope_deg",
    "min_valley_dist_km",
    "surge_height_m",
    "cumulative_precip_mm",
    "hazard_risk_level"  # Encoded ordinal
]

# Scenario Presets
SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "name": "Baseline Conditions",
        "name_hi": "सामान्य स्थिति (वर्तमान अवलोकन)",
        "rainfall_intensity_mmh": 12.0,
        "rainfall_duration_hrs": 6.0,
        "surge_height_m": 1.2,
        "upstream_breach_location": "Upper Bhote Koshi Glacier / Liping",
        "thermal_intensity_k": 2.5,
        "propagation_factor": 1.0,
        "road_closure_override": False,
        "description": "Observed post-event satellite conditions following the 26 Aug 2026 glacial surge."
    },
    "severe": {
        "name": "Severe Flood Scenario",
        "name_hi": "गंभीर बाढ़ परिदृश्य (+३.५ मी. जलवृद्धि)",
        "rainfall_intensity_mmh": 45.0,
        "rainfall_duration_hrs": 18.0,
        "surge_height_m": 3.8,
        "upstream_breach_location": "Upper Bhote Koshi Glacier / Liping",
        "thermal_intensity_k": 4.8,
        "propagation_factor": 1.45,
        "road_closure_override": True,
        "description": "Secondary moraine dam breach compounded by 18 hours of monsoon cloudburst (45 mm/h)."
    },
    "extreme": {
        "name": "Extreme Flood Scenario",
        "name_hi": "अत्यंत गंभीर बाढ़ परिदृश्य (+६.० मी. जलवृद्धि, तटबंध टूटना)",
        "rainfall_intensity_mmh": 85.0,
        "rainfall_duration_hrs": 36.0,
        "surge_height_m": 6.5,
        "upstream_breach_location": "Upper Bhote Koshi Glacier / Liping",
        "thermal_intensity_k": 6.8,
        "propagation_factor": 2.1,
        "road_closure_override": True,
        "description": "Catastrophic multi-lake cascading outburst flood with 85 mm/h extreme precipitation along the gorge."
    }
}

# Model Hyperparameters
RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 80,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": 1
}

GB_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.08,
    "max_depth": 5,
    "subsample": 0.85,
    "random_state": 42
}
