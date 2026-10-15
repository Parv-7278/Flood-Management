"""
FloodWatch-Himalaya: Synthetic & Pre-event OSM Sample Fixture Generator.
Creates georeferenced spatial vectors and raster arrays for August 2026
Trishuli / Bhote Koshi disaster corridor (Nepal) for offline validation and dashboard demos.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FIXTURES_DIR, CRS_WGS84, LEGAL_ATTRIBUTIONS


def create_trishuli_bhote_koshi_fixtures() -> Dict[str, Path]:
    """Generates realistic pre-event OSM vectors and EMSR927 reference polygons."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # 1. Settlements / Villages (Nodes)
    # Coordinates in WGS84 along the Bhote Koshi / Arniko Highway and Trishuli Highway
    settlements_data = [
        {"name": "Kodari", "name_ne": "कोदारी", "lat": 27.972, "lon": 85.968, "pop": 2400, "type": "village", "elevation_m": 1780},
        {"name": "Tatopani", "name_ne": "तातोपानी", "lat": 27.954, "lon": 85.938, "pop": 3100, "type": "village", "elevation_m": 1540},
        {"name": "Liping", "name_ne": "लिपिङ", "lat": 27.981, "lon": 85.975, "pop": 1150, "type": "hamlet", "elevation_m": 1820},
        {"name": "Larcha", "name_ne": "लार्चा", "lat": 27.925, "lon": 85.918, "pop": 1850, "type": "village", "elevation_m": 1460},
        {"name": "Chaku", "name_ne": "चाकु", "lat": 27.896, "lon": 85.892, "pop": 2100, "type": "village", "elevation_m": 1380},
        {"name": "Marming", "name_ne": "मार्मिङ", "lat": 27.882, "lon": 85.915, "pop": 1420, "type": "village", "elevation_m": 1620},
        {"name": "Hindi", "name_ne": "हिन्दी", "lat": 27.854, "lon": 85.881, "pop": 1600, "type": "village", "elevation_m": 1250},
        {"name": "Barhabise", "name_ne": "बाह्रबिसे", "lat": 27.789, "lon": 85.898, "pop": 9200, "type": "town", "elevation_m": 820},
        {"name": "Syaphrubesi", "name_ne": "स्याफ्रुबेँसी", "lat": 28.161, "lon": 85.342, "pop": 3400, "type": "town", "elevation_m": 1460},
        {"name": "Dhunche", "name_ne": "धुन्चे", "lat": 28.112, "lon": 85.301, "pop": 6800, "type": "district_capital", "elevation_m": 1960},
        {"name": "Mailung", "name_ne": "मैलुङ", "lat": 28.032, "lon": 85.253, "pop": 1550, "type": "village", "elevation_m": 1120},
        {"name": "Rasuwagadhi", "name_ne": "रसुवागढी", "lat": 28.272, "lon": 85.378, "pop": 950, "type": "border_post", "elevation_m": 1810},
    ]

    settlement_features = []
    for s in settlements_data:
        settlement_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
            "properties": s
        })

    settlements_geojson = {
        "type": "FeatureCollection",
        "name": "settlements",
        "features": settlement_features
    }
    settlements_path = FIXTURES_DIR / "settlements.geojson"
    with open(settlements_path, "w", encoding="utf-8") as f:
        json.dump(settlements_geojson, f, indent=2, ensure_ascii=False)
    paths["settlements"] = settlements_path

    # 2. Critical Facilities: Hospitals, Health Posts, Logistics Staging Hubs
    facilities_data = [
        {"name": "Dhunche District Hospital", "name_ne": "धुन्चे जिल्ला अस्पताल", "lat": 28.114, "lon": 85.298, "type": "hospital", "beds": 45, "helipad": True},
        {"name": "Barhabise Primary Healthcare Center", "name_ne": "बाह्रबिसे प्राथमिक स्वास्थ्य केन्द्र", "lat": 27.791, "lon": 85.901, "type": "hospital", "beds": 25, "helipad": True},
        {"name": "Tatopani Health Post", "name_ne": "तातोपानी स्वास्थ्य चौकी", "lat": 27.952, "lon": 85.935, "type": "health_post", "beds": 6, "helipad": False},
        {"name": "Syaphrubesi Health Center", "name_ne": "स्याफ्रुबेँसी स्वास्थ्य केन्द्र", "lat": 28.159, "lon": 85.340, "type": "clinic", "beds": 12, "helipad": True},
        {"name": "Sindhupalchok Emergency Ops Center", "name_ne": "सिन्धुपाल्चोक आपतकालीन सञ्चालन केन्द्र", "lat": 27.785, "lon": 85.892, "type": "staging_hub", "beds": 0, "helipad": True},
    ]
    facilities_features = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [fac["lon"], fac["lat"]]},
        "properties": fac
    } for fac in facilities_data]

    facilities_geojson = {"type": "FeatureCollection", "features": facilities_features}
    facilities_path = FIXTURES_DIR / "critical_facilities.geojson"
    with open(facilities_path, "w", encoding="utf-8") as f:
        json.dump(facilities_geojson, f, indent=2, ensure_ascii=False)
    paths["critical_facilities"] = facilities_path

    # 3. OSM Road Network (Pre-event: Arniko Highway & Pasang Lhamu Highway segments)
    # Connected segments forming the backbone graph
    road_segments = [
        # Bhote Koshi Corridor (Arniko Highway / H03)
        {"id": "road_01", "name": "Arniko Highway (Barhabise to Hindi)", "highway": "trunk", "ref": "H03", "length_m": 9400,
         "coords": [[85.898, 27.789], [85.889, 27.818], [85.881, 27.854]], "hazard_risk": "low"},
        {"id": "road_02", "name": "Arniko Highway (Hindi to Chaku)", "highway": "trunk", "ref": "H03", "length_m": 5800,
         "coords": [[85.881, 27.854], [85.887, 27.876], [85.892, 27.896]], "hazard_risk": "medium"},
        {"id": "road_03", "name": "Arniko Highway (Chaku to Larcha)", "highway": "trunk", "ref": "H03", "length_m": 4100,
         "coords": [[85.892, 27.896], [85.908, 27.912], [85.918, 27.925]], "hazard_risk": "high"},
        {"id": "road_04", "name": "Arniko Highway (Larcha to Tatopani)", "highway": "trunk", "ref": "H03", "length_m": 4300,
         "coords": [[85.918, 27.925], [85.929, 27.940], [85.938, 27.954]], "hazard_risk": "critical"},
        {"id": "road_05", "name": "Arniko Highway (Tatopani to Kodari)", "highway": "trunk", "ref": "H03", "length_m": 3900,
         "coords": [[85.938, 27.954], [85.952, 27.962], [85.968, 27.972]], "hazard_risk": "critical"},
        {"id": "road_06", "name": "Arniko Highway (Kodari to Liping Border)", "highway": "trunk", "ref": "H03", "length_m": 1500,
         "coords": [[85.968, 27.972], [85.975, 27.981]], "hazard_risk": "critical"},
        # Feeder mountain roads
        {"id": "road_feeder_marming", "name": "Marming Rural Feeder Road", "highway": "tertiary", "ref": "R01", "length_m": 3200,
         "coords": [[85.892, 27.896], [85.915, 27.882]], "hazard_risk": "high"},

        # Trishuli Corridor (Pasang Lhamu Highway / H21)
        {"id": "road_10", "name": "Pasang Lhamu Highway (Mailung to Dhunche)", "highway": "primary", "ref": "H21", "length_m": 12800,
         "coords": [[85.253, 28.032], [85.275, 28.072], [85.301, 28.112]], "hazard_risk": "medium"},
        {"id": "road_11", "name": "Pasang Lhamu Highway (Dhunche to Syaphrubesi)", "highway": "primary", "ref": "H21", "length_m": 8600,
         "coords": [[85.301, 28.112], [85.321, 28.138], [85.342, 28.161]], "hazard_risk": "medium"},
        {"id": "road_12", "name": "Pasang Lhamu Highway (Syaphrubesi to Rasuwagadhi)", "highway": "primary", "ref": "H21", "length_m": 16200,
         "coords": [[85.342, 28.161], [85.358, 28.212], [85.378, 28.272]], "hazard_risk": "high"},
    ]

    roads_features = [{
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": r["coords"]},
        "properties": {k: v for k, v in r.items() if k != "coords"}
    } for r in road_segments]

    roads_geojson = {"type": "FeatureCollection", "features": roads_features}
    roads_path = FIXTURES_DIR / "osm_highways_preevent.geojson"
    with open(roads_path, "w", encoding="utf-8") as f:
        json.dump(roads_geojson, f, indent=2, ensure_ascii=False)
    paths["roads"] = roads_path

    # 4. OSM Bridges (Pre-event)
    bridges_data = [
        {"id": "bridge_larcha", "name": "Larcha Steel Truss Bridge", "name_ne": "लार्चा पुल", "lat": 27.925, "lon": 85.918, "span_m": 72, "structure": "steel_truss"},
        {"id": "bridge_miteri", "name": "Friendship Bridge (Miteri Sanghu)", "name_ne": "मितेरी पुल", "lat": 27.981, "lon": 85.975, "span_m": 48, "structure": "concrete_girder"},
        {"id": "bridge_tatopani", "name": "Tatopani Gorge Crossing", "name_ne": "तातोपानी क्रसिङ", "lat": 27.954, "lon": 85.938, "span_m": 55, "structure": "suspension"},
        {"id": "bridge_chaku", "name": "Chaku Khola Bridge", "name_ne": "चाकु खोला पुल", "lat": 27.896, "lon": 85.892, "span_m": 38, "structure": "concrete_girder"},
        {"id": "bridge_syaphru", "name": "Syaphrubesi Trishuli Bridge", "name_ne": "स्याफ्रुबेँसी त्रिशुली पुल", "lat": 28.161, "lon": 85.342, "span_m": 64, "structure": "steel_truss"},
    ]
    bridges_features = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [b["lon"], b["lat"]]},
        "properties": b
    } for b in bridges_data]

    bridges_geojson = {"type": "FeatureCollection", "features": bridges_features}
    bridges_path = FIXTURES_DIR / "osm_bridges_preevent.geojson"
    with open(bridges_path, "w", encoding="utf-8") as f:
        json.dump(bridges_geojson, f, indent=2, ensure_ascii=False)
    paths["bridges"] = bridges_path

    # 5. OSM Buildings (Pre-event footprints in risk corridors)
    # Generate representative cluster polygons around settlements
    buildings_features = []
    building_idx = 1
    np.random.seed(42)

    cluster_centers = [
        (27.972, 85.968, 65, "Kodari"),
        (27.954, 85.938, 92, "Tatopani"),
        (27.981, 85.975, 34, "Liping"),
        (27.925, 85.918, 48, "Larcha"),
        (27.896, 85.892, 58, "Chaku"),
        (27.789, 85.898, 140, "Barhabise"),
        (28.161, 85.342, 85, "Syaphrubesi"),
        (28.112, 85.301, 160, "Dhunche"),
    ]

    for lat, lon, count, town in cluster_centers:
        for _ in range(count):
            d_lat = np.random.normal(0, 0.0025)
            d_lon = np.random.normal(0, 0.0025)
            b_lat = lat + d_lat
            b_lon = lon + d_lon
            size = np.random.uniform(0.00012, 0.00028)
            poly_coords = [
                [b_lon - size, b_lat - size],
                [b_lon + size, b_lat - size],
                [b_lon + size, b_lat + size],
                [b_lon - size, b_lat + size],
                [b_lon - size, b_lat - size]
            ]
            building_type = np.random.choice(["residential", "commercial", "industrial", "public"], p=[0.75, 0.15, 0.05, 0.05])
            buildings_features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [poly_coords]},
                "properties": {
                    "id": f"bldg_{building_idx:05d}",
                    "building": building_type,
                    "levels": int(np.random.choice([1, 2, 3], p=[0.4, 0.45, 0.15])),
                    "settlement": town
                }
            })
            building_idx += 1

    buildings_geojson = {"type": "FeatureCollection", "features": buildings_features}
    buildings_path = FIXTURES_DIR / "osm_buildings_preevent.geojson"
    with open(buildings_path, "w", encoding="utf-8") as f:
        json.dump(buildings_geojson, f, indent=2, ensure_ascii=False)
    paths["buildings"] = buildings_path

    # 6. Copernicus EMS EMSR927 Reference Vectors (Ground Truth Delineation)
    # Delineation of inundated river gorge and debris deposition corridor
    # Covering upper Bhote Koshi (Liping to below Larcha) and Mailung Trishuli flash zone
    emsr_polygons = [
        # Bhote Koshi high torrent & debris corridor
        {
            "id": "EMSR927_DELIN_01",
            "event_type": "Glacier Collapse & Hyperconcentrated Debris Flow",
            "sensor": "Sentinel-1A + Sentinel-2A / WorldView-3",
            "valid_date": "2026-08-27T06:00:00Z",
            "area_sqkm": 8.45,
            "coords": [
                [85.910, 27.915],
                [85.918, 27.923],
                [85.922, 27.929],
                [85.932, 27.942],
                [85.940, 27.956],
                [85.955, 27.965],
                [85.972, 27.975],
                [85.980, 27.985],
                [85.975, 27.988],
                [85.962, 27.978],
                [85.946, 27.968],
                [85.932, 27.952],
                [85.923, 27.938],
                [85.912, 27.922],
                [85.905, 27.915],
                [85.910, 27.915]
            ]
        },
        # Chaku-Marming tributary blockage impoundment lake
        {
            "id": "EMSR927_DELIN_02",
            "event_type": "Debris Dammed Impoundment Lake",
            "sensor": "Sentinel-2B Optical MNDWI",
            "valid_date": "2026-08-27T08:30:00Z",
            "area_sqkm": 2.15,
            "coords": [
                [85.888, 27.890],
                [85.897, 27.893],
                [85.899, 27.901],
                [85.892, 27.904],
                [85.885, 27.898],
                [85.888, 27.890]
            ]
        },
        # Trishuli - Mailung Gorge Flash Flood segment
        {
            "id": "EMSR927_DELIN_03",
            "event_type": "Flash Flood & Bank Scour",
            "sensor": "Sentinel-1B IW GRD",
            "valid_date": "2026-08-27T06:00:00Z",
            "area_sqkm": 3.75,
            "coords": [
                [85.248, 28.025],
                [85.258, 28.035],
                [85.265, 28.050],
                [85.258, 28.054],
                [85.250, 28.040],
                [85.242, 28.028],
                [85.248, 28.025]
            ]
        }
    ]

    emsr_features = [{
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [poly["coords"]]},
        "properties": {k: v for k, v in poly.items() if k != "coords"}
    } for poly in emsr_polygons]

    emsr_geojson = {
        "type": "FeatureCollection",
        "name": "EMSR927_Delineation_GroundTruth",
        "attribution": LEGAL_ATTRIBUTIONS["emsr_reference"],
        "features": emsr_features
    }
    emsr_path = FIXTURES_DIR / "emsr927_ground_truth.geojson"
    with open(emsr_path, "w", encoding="utf-8") as f:
        json.dump(emsr_geojson, f, indent=2, ensure_ascii=False)
    paths["emsr_ground_truth"] = emsr_path

    return paths


if __name__ == "__main__":
    generated = create_trishuli_bhote_koshi_fixtures()
    print("Successfully generated all sample geospatial fixtures:")
    for k, p in generated.items():
        print(f"  - {k}: {p}")
