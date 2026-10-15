"""
FastAPI REST API Integration Tests for Thermal Analysis Endpoints.
Verifies /api/thermal/presets, /api/thermal/available, /api/thermal/process,
/api/thermal/status, /api/thermal/metadata, /api/thermal/layer, and /api/thermal/point.
"""

import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app


class TestThermalAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_get_presets(self):
        """Verifies geographical AOI presets endpoint."""
        resp = self.client.get("/api/thermal/presets")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("presets", data)
        self.assertIn("Bhote Koshi Gorge Corridor (Liping to Barhabise)", data["presets"])
        self.assertIn("default", data)

    def test_02_search_available_acquisitions(self):
        """Verifies acquisition search endpoint."""
        resp = self.client.get("/api/thermal/available?satellite=landsat-c2-l2&limit=2")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_found", data)
        self.assertGreater(data["total_found"], 0)
        self.assertIn("acquisitions", data)
        first = data["acquisitions"][0]
        self.assertIn("id", first)
        self.assertIn("platform", first)

    def test_03_process_surface_temperature(self):
        """Verifies surface temperature processing mode."""
        payload = {
            "mode": "surface_temp",
            "satellite": "landsat-c2-l2"
        }
        resp = self.client.post("/api/thermal/process", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "COMPLETED")
        self.assertEqual(data["mode"], "surface_temp")
        self.assertIn("stats", data)
        self.assertIn("mean_celsius", data["stats"])
        self.assertIn("legend", data)
        self.assertEqual(data["legend"]["unit"], "°C")

    def test_04_process_temperature_anomaly(self):
        """Verifies pre vs post anomaly delta processing mode."""
        payload = {
            "mode": "anomaly",
            "satellite": "landsat-c2-l2"
        }
        resp = self.client.post("/api/thermal/process", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "COMPLETED")
        self.assertEqual(data["mode"], "anomaly")
        self.assertIn("stats", data)
        self.assertGreater(data["stats"]["max_positive_anomaly_k"], 3.0)

        job_id = data["job_id"]

        # Check metadata
        meta_resp = self.client.get(f"/api/thermal/metadata/{job_id}")
        self.assertEqual(meta_resp.status_code, 200)
        meta = meta_resp.json()
        self.assertEqual(meta["mode"], "anomaly")

        # Check layer
        layer_resp = self.client.get(f"/api/thermal/layer/{job_id}")
        self.assertEqual(layer_resp.status_code, 200)
        layer = layer_resp.json()
        self.assertTrue(layer["overlay_uri"].startswith("data:image/png;base64,"))

        # Check point probe at Tatopani (27.954 N, 85.938 E)
        pt_resp = self.client.get(f"/api/thermal/point?job_id={job_id}&lat=27.954&lon=85.938")
        self.assertEqual(pt_resp.status_code, 200)
        pt = pt_resp.json()
        self.assertIsNotNone(pt["value"])
        self.assertGreater(pt["value"], 2.0)

    def test_05_nonexistent_job_returns_404(self):
        """Verifies 404 response for invalid job_id."""
        resp = self.client.get("/api/thermal/metadata/nonexistent_job_12345")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
