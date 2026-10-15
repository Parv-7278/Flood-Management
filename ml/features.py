"""
FloodWatch-Himalaya: Machine Learning Feature Engineering Pipeline.
Extracts topographic, meteorological, hydrological, thermal, and exposure features
for geographic grid cells and road segments across the Trishuli/Bhote Koshi basin.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import FEATURE_NAMES, ROAD_FEATURE_NAMES


class FeaturePipeline:
    """
    Constructs multimodal geospatial feature vectors for predictive flood modeling.
    Combines Copernicus DEM-30 topography, river geometry, meteorological forcing,
    kinetic thermal anomaly proxies, and infrastructure exposure.
    """

    @staticmethod
    def compute_topographic_features(
        elevation_grid: np.ndarray,
        valley_dist_grid: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Derives local slope, aspect, and flow accumulation proxy from elevation and thalweg distance.
        """
        dy, dx = np.gradient(elevation_grid, 30.0, 30.0)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)

        # Flow accumulation proxy: high along valley bottom and drainage channels
        flow_accum_proxy = np.clip(1.0 / (valley_dist_grid + 0.005), 1.0, 200.0)

        return {
            "elevation_m": elevation_grid,
            "slope_deg": slope_deg,
            "valley_dist_km": valley_dist_grid * 111.0,  # Convert degrees to approx km
            "flow_accumulation_proxy": flow_accum_proxy
        }

    @staticmethod
    def generate_training_dataset(
        num_samples_per_sector: int = 1200,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Synthesizes a geophysically grounded training dataset simulating diverse flood events
        across 4 distinct geographic sectors in the Trishuli and Bhote Koshi basins:
        - Sector 1: Upper Bhote Koshi (Liping - Kodari - Tatopani high gorge)
        - Sector 2: Middle Bhote Koshi (Larcha - Chaku - Hindi steep canyon)
        - Sector 3: Lower Bhote Koshi (Barhabise confluence and valley floor)
        - Sector 4: Trishuli Valley (Syaphrubesi - Mailung - Dhunche corridor)

        Enables geographic train/validation/test splits to test cross-basin generalization.
        """
        np.random.seed(random_seed)
        records: List[Dict[str, Any]] = []

        sectors = [
            {"sector_id": "bhote_upper", "elev_range": (1500, 2600), "slope_mean": 38.0, "valley_narrow": True},
            {"sector_id": "bhote_middle", "elev_range": (1200, 2100), "slope_mean": 32.0, "valley_narrow": True},
            {"sector_id": "bhote_lower", "elev_range": (800, 1600), "slope_mean": 22.0, "valley_narrow": False},
            {"sector_id": "trishuli_west", "elev_range": (1100, 2400), "slope_mean": 35.0, "valley_narrow": True},
        ]

        for s in sectors:
            for _ in range(num_samples_per_sector):
                # Topography
                elev_min, elev_max = s["elev_range"]
                elevation_m = np.random.uniform(elev_min, elev_max)
                slope_deg = np.clip(np.random.normal(s["slope_mean"], 8.0), 2.0, 75.0)

                # Distance from river thalweg (km)
                valley_dist_km = np.random.exponential(scale=1.2 if s["valley_narrow"] else 2.5)

                # Flow accumulation proxy
                flow_accum = np.clip(250.0 / (valley_dist_km + 0.1), 5.0, 500.0)

                # Hydrometeorological forcing
                rainfall_intensity = np.random.uniform(5.0, 95.0)  # mm/h
                rainfall_duration = np.random.uniform(2.0, 48.0)    # hrs
                cumulative_precip = rainfall_intensity * rainfall_duration

                # River surge height (meters above baseflow)
                # Driven by upstream moraine breach + precipitation accumulation
                surge_height = np.clip(
                    np.random.uniform(0.5, 7.5) + (cumulative_precip / 800.0),
                    0.5, 9.0
                )
                upstream_discharge = 150.0 + (surge_height ** 2.2) * 45.0 + rainfall_intensity * 8.0

                # Thermal anomaly (Kelvin) - friction kinetic heating in rapid slurry
                thermal_anomaly = np.clip(
                    (surge_height * 0.7) + np.random.normal(0.8, 0.4),
                    0.0, 8.5
                )
                propagation_factor = np.clip(0.8 + (surge_height / 5.0), 0.8, 2.5)

                # Exposure features
                building_density = np.random.exponential(scale=35.0 if not s["valley_narrow"] else 15.0)
                road_proximity = np.clip(np.random.exponential(scale=250.0), 5.0, 3000.0)
                bridge_proximity = np.clip(np.random.exponential(scale=1200.0), 20.0, 8000.0)

                # Physical Hydrodynamic Inundation Probability Formulation (Ground Truth Target)
                # Water spills onto terraces when surge exceeds channel bank height (~2.5m)
                # Inundation risk drops with elevation difference from thalweg (proportional to valley_dist_km * tan(slope))
                effective_height_above_river = valley_dist_km * 1000.0 * np.sin(np.radians(max(2.0, slope_deg * 0.35)))
                
                hydro_risk_score = (
                    (surge_height * 3.8)
                    - (effective_height_above_river * 0.18)
                    + (cumulative_precip / 150.0)
                    - (valley_dist_km * 4.5)
                    + (flow_accum / 80.0)
                )

                # Probability via sigmoid
                inundation_prob = 1.0 / (1.0 + np.exp(-hydro_risk_score))
                is_inundated = int(inundation_prob >= 0.50)

                records.append({
                    "sector_id": s["sector_id"],
                    "elevation_m": elevation_m,
                    "slope_deg": slope_deg,
                    "valley_dist_km": valley_dist_km,
                    "flow_accumulation_proxy": flow_accum,
                    "rainfall_intensity_mmh": rainfall_intensity,
                    "rainfall_duration_hrs": rainfall_duration,
                    "cumulative_precip_mm": cumulative_precip,
                    "surge_height_m": surge_height,
                    "upstream_discharge_m3s": upstream_discharge,
                    "thermal_anomaly_k": thermal_anomaly,
                    "propagation_factor": propagation_factor,
                    "building_density_sqkm": building_density,
                    "road_proximity_m": road_proximity,
                    "bridge_proximity_m": bridge_proximity,
                    "inundation_prob_true": inundation_prob,
                    "is_inundated": is_inundated
                })

        df = pd.DataFrame(records)
        return df

    @staticmethod
    def generate_road_training_dataset(num_samples: int = 800) -> pd.DataFrame:
        """
        Synthesizes a road segment damage dataset for training regression models
        predicting meters of submerged/blocked highway.
        """
        np.random.seed(101)
        records: List[Dict[str, Any]] = []

        risk_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}

        for _ in range(num_samples):
            length_m = np.random.uniform(500.0, 16000.0)
            mean_elevation_m = np.random.uniform(900.0, 2100.0)
            mean_slope_deg = np.random.uniform(8.0, 45.0)
            min_valley_dist_km = np.random.exponential(scale=0.8)
            surge_height_m = np.random.uniform(0.5, 8.5)
            cumulative_precip_mm = np.random.uniform(40.0, 1200.0)
            risk_cat = np.random.choice(["low", "medium", "high", "critical"], p=[0.25, 0.35, 0.25, 0.15])
            risk_encoded = risk_levels[risk_cat]

            # Inundation fraction based on surge, valley proximity, and precipitation
            inundation_fraction = np.clip(
                (surge_height_m / 9.0) * 0.65
                + (1.0 / (min_valley_dist_km + 0.2)) * 0.25
                + (cumulative_precip_mm / 1500.0) * 0.20
                - (mean_elevation_m / 6000.0),
                0.0, 1.0
            )
            flooded_length_m = round(length_m * inundation_fraction, 1)

            records.append({
                "length_m": length_m,
                "mean_elevation_m": mean_elevation_m,
                "mean_slope_deg": mean_slope_deg,
                "min_valley_dist_km": min_valley_dist_km,
                "surge_height_m": surge_height_m,
                "cumulative_precip_mm": cumulative_precip_mm,
                "hazard_risk_level": risk_encoded,
                "flooded_length_m": flooded_length_m
            })

        return pd.DataFrame(records)
