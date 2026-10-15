"""
FloodWatch-Himalaya: Multimodal AI Hackathon 2026 (Track B: Mapping Flood Damage from Space)
Configuration and System Constants for August 2026 Trishuli/Bhote Koshi Disaster.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIXTURES_DIR = DATA_DIR / "sample_fixtures"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
FIXTURES_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Area of Interest (AOI): Trishuli / Bhote Koshi River Basin, Bagmati Province, Nepal
# High-relief Himalayan corridor connecting Syaphrubesi, Dhunche, Rasuwagadhi, Tatopani, Kodari
AOI_BBOX_WGS84: Tuple[float, float, float, float] = (
    85.25,  # min_lon (west)
    27.85,  # min_lat (south)
    86.05,  # max_lon (east)
    28.38   # max_lat (north)
)

AOI_CENTROID_LAT = 28.05
AOI_CENTROID_LON = 85.65

# Projection systems
CRS_WGS84 = "EPSG:4326"
CRS_UTM_NEPAL = "EPSG:32645"  # WGS 84 / UTM Zone 45N (standard for central/eastern Nepal)

# Event Metadata
DISASTER_EVENT_NAME = "August 2026 Trishuli / Bhote Koshi Glacier-Collapse Flash Flood"
DISASTER_EVENT_DATE = "2026-08-26"
PRE_EVENT_CUTOFF_TIMESTAMP = "2026-08-25T23:59:59Z"
COPERNICUS_EMSR_CODE = "EMSR927"

# Sentinel-1 SAR Parameters
SAR_SPECKLE_FILTER_WINDOW = 7
SAR_LOOKS_ENL = 4.4  # Sentinel-1 IW GRD equivalent number of looks
SAR_FLOOD_DB_DIFF_THRESHOLD = -3.2   # Negative backscatter anomaly for standing/smooth flood water (dB)
SAR_DEBRIS_DB_DIFF_THRESHOLD = 3.5   # Positive backscatter anomaly for rough rubbly boulder debris (dB)
SAR_RELATIVE_ORBIT_TRACK = 121       # Descending track for Bhote Koshi gorge consistency

# Sentinel-2 Optical Parameters
MNDWI_WATER_THRESHOLD = 0.12        # Modified NDWI threshold for turbid mountain water
S2_CLOUD_PROBABILITY_THRESHOLD = 0.35

# Landsat-8/9 & Sentinel-3 Thermal LST Parameters
# Debris flow friction kinetic heating / warm impoundment anomaly threshold in Kelvin
THERMAL_ANOMALY_DELTA_KELVIN_THRESHOLD = 2.8

# Mandatory Text Attributions for deliverables and outputs
LEGAL_ATTRIBUTIONS: Dict[str, str] = {
    "copernicus_sentinel": "Contains modified Copernicus Sentinel data [2026], processed by FloodWatch-Himalaya.",
    "copernicus_dem": "Produced using Copernicus WorldDEM-30 © DLR e.V. 2026, European Union.",
    "osm_ohsome": "OSM data © OpenStreetMap contributors, accessed via HeiGIT ohsome API (snapshot < 2026-08-26).",
    "emsr_reference": "Reference validation derived from Copernicus Emergency Management Service [EMSR927]."
}

OHSOME_API_ENDPOINT = "https://api.ohsome.org/v1"

if __name__ == "__main__":
    print("=" * 70)
    print("🛰️  FLOODWATCH-HIMALAYA SYSTEM CONFIGURATION")
    print("=" * 70)
    print(f"Disaster Event:   {DISASTER_EVENT_NAME}")
    print(f"Event Date:       {DISASTER_EVENT_DATE}")
    print(f"Copernicus Code:  {COPERNICUS_EMSR_CODE}")
    print(f"AOI BBox (WGS84): {AOI_BBOX_WGS84}")
    print(f"AOI Centroid:     ({AOI_CENTROID_LAT}°N, {AOI_CENTROID_LON}°E)")
    print(f"Coordinate Ref:   {CRS_WGS84} -> {CRS_UTM_NEPAL}")
    print(f"Data Directory:   {DATA_DIR}")
    print(f"Cache Directory:  {CACHE_DIR}")
    print("=" * 70)
