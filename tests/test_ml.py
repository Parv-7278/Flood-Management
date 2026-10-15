"""
Unit & Integration Tests for FloodWatch-Himalaya AI/ML Prediction Pipeline.
Validates model loading, feature validation, preset predictability, and NetworkX isolation routing.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import SCENARIO_PRESETS, FLOOD_CLASSIFIER_PATH, ROAD_REGRESSOR_PATH
from ml.predict import WhatIfFloodPredictor
from ml.features import FeaturePipeline


class TestMLPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = WhatIfFloodPredictor()

    def test_model_binaries_exist(self):
        """Verifies that trained serialized models exist on disk."""
        self.assertTrue(FLOOD_CLASSIFIER_PATH.exists(), "Flood classifier binary missing.")
        self.assertTrue(ROAD_REGRESSOR_PATH.exists(), "Road regressor binary missing.")

    def test_training_dataset_generation(self):
        """Verifies feature dataset generation and column schema."""
        df = FeaturePipeline.generate_training_dataset(num_samples_per_sector=50)
        self.assertGreater(len(df), 150)
        self.assertIn("elevation_m", df.columns)
        self.assertIn("surge_height_m", df.columns)
        self.assertIn("is_inundated", df.columns)

    def test_baseline_preset_simulation(self):
        """Verifies execution of Baseline scenario preset."""
        preset = SCENARIO_PRESETS["baseline"]
        res = self.predictor.predict_scenario(preset)
        self.assertIn("predictions", res)
        self.assertIn("comparison_with_baseline", res)
        p = res["predictions"]
        self.assertGreaterEqual(p["inundated_area_sqkm"], 0.0)
        self.assertGreaterEqual(p["impacted_buildings_estimate"], 0)
        self.assertGreater(p["flooded_road_length_meters"], 0.0)

    def test_extreme_preset_increases_severity(self):
        """Verifies that Extreme scenario predicts greater impact than Baseline."""
        base_res = self.predictor.predict_scenario(SCENARIO_PRESETS["baseline"])
        ext_res = self.predictor.predict_scenario(SCENARIO_PRESETS["extreme"])

        base_p = base_res["predictions"]
        ext_p = ext_res["predictions"]

        self.assertGreaterEqual(ext_p["impacted_buildings_estimate"], base_p["impacted_buildings_estimate"])
        self.assertGreaterEqual(ext_p["cutoff_settlement_count"], base_p["cutoff_settlement_count"])

    def test_confidence_bounds(self):
        """Verifies confidence interval ordering."""
        res = self.predictor.predict_scenario(SCENARIO_PRESETS["severe"])
        ci = res["predictions"]["impacted_buildings_ci"]
        self.assertLessEqual(ci[0], ci[1])


if __name__ == "__main__":
    unittest.main()
