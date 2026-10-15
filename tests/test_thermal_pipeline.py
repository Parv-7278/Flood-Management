"""
Unit & Integration Tests for Spaceborne Thermal Analysis Pipeline.
Verifies STAC catalog search, Landsat C2 L2 physical calibration,
quality masking, spatial reprojection, anomaly detection, and point probing.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.thermal_pipeline import (
    ThermalDataService,
    ThermalRasterProcessor,
    THERMAL_AOI_PRESETS,
    LANDSAT_ST_SCALE,
    LANDSAT_ST_OFFSET,
    KELVIN_TO_CELSIUS
)
from config import FIXTURES_DIR


class TestThermalPipeline(unittest.TestCase):

    def test_01_physical_calibration_math(self):
        """Verifies Landsat Collection 2 Level-2 surface temperature scaling formula."""
        # For raw DN = 40000:
        # Kelvin = 40000 * 0.00341802 + 149.0 = 285.7208 K
        # Celsius = 285.7208 - 273.15 = 12.5708 °C
        raw_dn = 40000
        expected_kelvin = raw_dn * LANDSAT_ST_SCALE + LANDSAT_ST_OFFSET
        expected_celsius = expected_kelvin - KELVIN_TO_CELSIUS
        self.assertAlmostEqual(expected_kelvin, 285.7208, places=3)
        self.assertAlmostEqual(expected_celsius, 12.5708, places=3)

    def test_02_stac_search_and_fallback(self):
        """Verifies STAC discovery over Microsoft Planetary Computer with reliable fallback."""
        items = ThermalDataService.search_acquisitions(
            start_date="2024-05-01",
            end_date="2024-05-30",
            limit=2
        )
        self.assertGreater(len(items), 0)
        first_item = items[0]
        self.assertIn("id", first_item)
        self.assertIn("date", first_item)
        self.assertIn("platform", first_item)
        self.assertIn("cloud_cover", first_item)

    def test_03_load_surface_temperature_reference_fixtures(self):
        """Verifies raster loading, window clipping, and degree Celsius conversion."""
        pre_fixture = FIXTURES_DIR / "landsat_pre_event_lst_c2l2.tif"
        self.assertTrue(pre_fixture.exists(), "Pre-event fixture must exist")

        res = ThermalRasterProcessor.load_surface_temperature(str(pre_fixture))
        self.assertEqual(len(res["shape"]), 2)
        self.assertIn("stats", res)
        stats = res["stats"]
        self.assertGreater(stats["valid_pixels"], 0)
        # Normal Himalayan temperatures should be within reasonable bounds (-25°C to 45°C)
        self.assertGreater(stats["min_celsius"], -25.0)
        self.assertLess(stats["max_celsius"], 45.0)
        self.assertGreater(stats["mean_celsius"], -10.0)

    def test_04_thermal_anomaly_calculation(self):
        """Verifies pre vs post anomaly delta calculation: ΔT = T_post - T_pre."""
        pre_fixture = FIXTURES_DIR / "landsat_pre_event_lst_c2l2.tif"
        post_fixture = FIXTURES_DIR / "landsat_post_event_lst_c2l2.tif"

        pre_res = ThermalRasterProcessor.load_surface_temperature(str(pre_fixture))
        post_res = ThermalRasterProcessor.load_surface_temperature(str(post_fixture))

        anomaly = ThermalRasterProcessor.compute_anomaly(pre_res, post_res)
        self.assertIn("delta_temp", anomaly)
        self.assertIn("stats", anomaly)
        stats = anomaly["stats"]

        # Post-event friction heating must exhibit positive anomaly >= 3.0 K
        self.assertGreater(stats["max_positive_anomaly_k"], 3.0)
        self.assertGreater(stats["friction_heated_pixels"], 0)
        self.assertGreater(stats["friction_heated_area_km2"], 0.0)

    def test_05_overlay_png_generation(self):
        """Verifies transparent RGBA PNG generation for Leaflet image overlay."""
        pre_fixture = FIXTURES_DIR / "landsat_pre_event_lst_c2l2.tif"
        res = ThermalRasterProcessor.load_surface_temperature(str(pre_fixture))

        data_uri, legend = ThermalRasterProcessor.render_overlay_png(res["temp_celsius"], mode="surface_temp")
        self.assertTrue(data_uri.startswith("data:image/png;base64,"))
        self.assertIn("vmin", legend)
        self.assertIn("vmax", legend)
        self.assertEqual(legend["unit"], "°C")

    def test_06_point_temperature_probe(self):
        """Verifies geographic point probe returns valid temperature at known coordinate."""
        pre_fixture = FIXTURES_DIR / "landsat_pre_event_lst_c2l2.tif"
        res = ThermalRasterProcessor.load_surface_temperature(str(pre_fixture))

        # Sample at Tatopani (27.954 N, 85.938 E)
        temp_val = ThermalRasterProcessor.sample_point_temperature(
            res["temp_celsius"],
            res["transform"],
            res["crs"],
            lat=27.954,
            lon=85.938
        )
        self.assertIsNotNone(temp_val)
        self.assertIsInstance(temp_val, float)
        self.assertGreater(temp_val, -15.0)
        self.assertLess(temp_val, 40.0)


if __name__ == "__main__":
    unittest.main()
