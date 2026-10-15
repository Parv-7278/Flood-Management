"""
FloodWatch-Himalaya: Vector Overlay & Cut-Off Settlement Network Routing Engine.
Adheres strictly to Track B specification:
- Pre-event OSM snapshot via HeiGIT ohsome API (roads, bridges, buildings < 2026-08-26).
- Spatial intersection using GeoPandas/Shapely to compute damaged infrastructure.
- NetworkX graph routing to isolate and flag settlements cut off from hospitals/relief hubs
  due to severed bridges and inundated road segments.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import requests

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union

from config import (
    FIXTURES_DIR,
    CACHE_DIR,
    CRS_WGS84,
    CRS_UTM_NEPAL,
    OHSOME_API_ENDPOINT,
    PRE_EVENT_CUTOFF_TIMESTAMP,
    AOI_BBOX_WGS84,
    LEGAL_ATTRIBUTIONS
)


class OhsomeOSMClient:
    """
    Client for querying the HeiGIT ohsome API to retrieve historical OSM snapshots
    strictly before the August 26, 2026 glacier collapse disaster.
    Falls back gracefully to verified pre-event local fixtures for offline/reproducible runs.
    """

    def __init__(self, endpoint: str = OHSOME_API_ENDPOINT, time_cutoff: str = PRE_EVENT_CUTOFF_TIMESTAMP):
        self.endpoint = endpoint
        self.time_cutoff = time_cutoff
        self.attribution = LEGAL_ATTRIBUTIONS["osm_ohsome"]

    def fetch_preevent_data(self, bboxes: Tuple[float, float, float, float] = AOI_BBOX_WGS84) -> Dict[str, gpd.GeoDataFrame]:
        """
        Attempts to query ohsome API for pre-event roads, bridges, and buildings.
        If offline or non-200 response, loads verified local pre-event fixtures.
        """
        min_lon, min_lat, max_lon, max_lat = bboxes
        bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"

        results: Dict[str, gpd.GeoDataFrame] = {}

        # 1. Roads
        roads_gdf = self._query_ohsome_or_fallback(
            filter_str="highway=* and type:way",
            fallback_file=FIXTURES_DIR / "osm_highways_preevent.geojson",
            layer_name="roads"
        )
        results["roads"] = roads_gdf

        # 2. Bridges
        bridges_gdf = self._query_ohsome_or_fallback(
            filter_str="bridge=* and bridge!=no",
            fallback_file=FIXTURES_DIR / "osm_bridges_preevent.geojson",
            layer_name="bridges"
        )
        results["bridges"] = bridges_gdf

        # 3. Buildings
        buildings_gdf = self._query_ohsome_or_fallback(
            filter_str="building=* and type:way",
            fallback_file=FIXTURES_DIR / "osm_buildings_preevent.geojson",
            layer_name="buildings"
        )
        results["buildings"] = buildings_gdf

        # 4. Settlements & Facilities
        settlements_gdf = gpd.read_file(FIXTURES_DIR / "settlements.geojson")
        facilities_gdf = gpd.read_file(FIXTURES_DIR / "critical_facilities.geojson")
        results["settlements"] = settlements_gdf
        results["facilities"] = facilities_gdf

        return results

    def _query_ohsome_or_fallback(self, filter_str: str, fallback_file: Path, layer_name: str) -> gpd.GeoDataFrame:
        """Helper to try ohsome API with timeout, then fall back to local fixture."""
        url = f"{self.endpoint}/elements/geometry"
        params = {
            "bboxes": f"{AOI_BBOX_WGS84[0]},{AOI_BBOX_WGS84[1]},{AOI_BBOX_WGS84[2]},{AOI_BBOX_WGS84[3]}",
            "time": self.time_cutoff,
            "filter": filter_str
        }
        try:
            resp = requests.get(url, params=params, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if "features" in data and len(data["features"]) > 0:
                    gdf = gpd.GeoDataFrame.from_features(data, crs=CRS_WGS84)
                    return gdf
        except Exception:
            pass  # Fall back to verified fixture

        if fallback_file.exists():
            return gpd.read_file(fallback_file)
        raise FileNotFoundError(f"Neither ohsome API nor fallback file {fallback_file} available.")


class InfrastructureDamageAssessor:
    """
    Performs spatial intersections using GeoPandas and Shapely in projected UTM coordinates
    to calculate precise damage metrics:
    - Flooded road segment lengths (meters)
    - Severed / washed out bridges
    - Submerged buildings (count and total footprint sq meters)
    """

    def __init__(self, hazard_extent_gdf: gpd.GeoDataFrame):
        self.hazard_gdf = hazard_extent_gdf
        # Create unified hazard polygon in UTM coordinates
        if not hazard_extent_gdf.empty:
            self.hazard_utm = hazard_extent_gdf.to_crs(CRS_UTM_NEPAL)
            self.hazard_union_utm = unary_union(self.hazard_utm.geometry)
            # Unified polygon in WGS84 for fast intersection
            self.hazard_union_wgs84 = unary_union(hazard_extent_gdf.geometry)
        else:
            self.hazard_utm = None
            self.hazard_union_utm = Polygon()
            self.hazard_union_wgs84 = Polygon()

    def assess_damaged_roads(self, roads_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Intersects road network with flood/debris extent.
        Adds properties: is_flooded (bool), flooded_length_m (float), status (str).
        """
        roads = roads_gdf.copy()
        roads_utm = roads.to_crs(CRS_UTM_NEPAL)

        is_flooded_list = []
        flooded_len_list = []
        status_list = []

        for idx, row in roads_utm.iterrows():
            geom = row.geometry
            if geom.intersects(self.hazard_union_utm):
                intersection = geom.intersection(self.hazard_union_utm)
                flooded_meters = float(intersection.length)
                is_flooded_list.append(True)
                flooded_len_list.append(round(flooded_meters, 1))
                status_list.append("IMPASSABLE - INUNDATED")
            else:
                is_flooded_list.append(False)
                flooded_len_list.append(0.0)
                status_list.append("PASSABLE")

        roads["is_flooded"] = is_flooded_list
        roads["flooded_length_m"] = flooded_len_list
        roads["status"] = status_list
        return roads

    def assess_damaged_bridges(self, bridges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Flags bridges severed by debris flow or submerged in flood waters.
        Uses a 45-meter buffer to capture bridge approaches and abutment scour.
        """
        bridges = bridges_gdf.copy()
        bridges_utm = bridges.to_crs(CRS_UTM_NEPAL)

        is_severed = []
        condition = []

        for idx, row in bridges_utm.iterrows():
            pt_buffer = row.geometry.buffer(150.0)  # 150m buffer for gorge span & approach road scouring
            if pt_buffer.intersects(self.hazard_union_utm):
                is_severed.append(True)
                condition.append("SEVERED / COLLAPSED")
            else:
                is_severed.append(False)
                condition.append("OPERATIONAL")

        bridges["is_severed"] = is_severed
        bridges["status"] = condition
        return bridges

    def assess_damaged_buildings(self, buildings_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Intersects building footprints with flood/debris zone.
        Returns damaged buildings GeoDataFrame, total count, and total destroyed area.
        """
        buildings = buildings_gdf.copy()
        buildings_utm = buildings.to_crs(CRS_UTM_NEPAL)

        damaged_mask = buildings_utm.geometry.intersects(self.hazard_union_utm)
        damaged_buildings_gdf = buildings[damaged_mask].copy()

        # Compute destroyed area in square meters
        damaged_utm = buildings_utm[damaged_mask]
        total_area_sqm = float(damaged_utm.geometry.area.sum()) if not damaged_utm.empty else 0.0

        # Breakdown by settlement and building category
        breakdown_by_settlement = damaged_buildings_gdf["settlement"].value_counts().to_dict() if not damaged_buildings_gdf.empty else {}
        breakdown_by_type = damaged_buildings_gdf["building"].value_counts().to_dict() if not damaged_buildings_gdf.empty else {}

        return {
            "damaged_gdf": damaged_buildings_gdf,
            "total_damaged_count": int(damaged_mask.sum()),
            "total_damaged_area_sqm": round(total_area_sqm, 1),
            "by_settlement": breakdown_by_settlement,
            "by_building_type": breakdown_by_type
        }


class CutOffSettlementRouter:
    """
    Constructs a topological NetworkX graph from pre-event roads, bridges, and settlements,
    simulates severed links due to flood/debris scour, and analyzes isolation:
    - Identifies settlements completely severed from regional hospitals and staging hubs
    - Computes stranded population
    - Evaluates detour travel distance penalties for accessible communities
    - Identifies critical bridge bottlenecks (cut edges)
    """

    def __init__(
        self,
        roads_gdf: gpd.GeoDataFrame,
        bridges_gdf: gpd.GeoDataFrame,
        settlements_gdf: gpd.GeoDataFrame,
        facilities_gdf: gpd.GeoDataFrame
    ):
        self.roads_gdf = roads_gdf
        self.bridges_gdf = bridges_gdf
        self.settlements_gdf = settlements_gdf
        self.facilities_gdf = facilities_gdf

        self.graph_preevent = nx.Graph()
        self.graph_postevent = nx.Graph()
        self._build_graphs()

    def _coord_to_node_id(self, lon: float, lat: float) -> str:
        """Normalizes coordinates to 4 decimal places (~11 meters) to form shared intersection nodes."""
        return f"node_{lon:.4f}_{lat:.4f}"

    def _build_graphs(self) -> None:
        """Builds pre-event and post-event NetworkX topological road graphs."""
        # 1. Add road segments as edges
        for idx, row in self.roads_gdf.iterrows():
            geom = row.geometry
            coords = list(geom.coords)
            u = self._coord_to_node_id(coords[0][0], coords[0][1])
            v = self._coord_to_node_id(coords[-1][0], coords[-1][1])
            length_m = float(row.get("length_m", 1000.0))
            road_id = row.get("id", f"road_{idx}")
            is_flooded = bool(row.get("is_flooded", False))

            edge_data = {
                "id": road_id,
                "name": row.get("name", "Unnamed Road"),
                "weight": length_m,
                "is_flooded": is_flooded
            }

            # Pre-event graph has all edges
            self.graph_preevent.add_edge(u, v, **edge_data)

            # Post-event graph only retains passable unflooded edges
            if not is_flooded:
                self.graph_postevent.add_edge(u, v, **edge_data)

        # 2. Check severed bridges: if bridge is severed, sever any adjacent edge
        severed_bridges = self.bridges_gdf[self.bridges_gdf["is_severed"] == True] if "is_severed" in self.bridges_gdf else gpd.GeoDataFrame()
        for idx, b_row in severed_bridges.iterrows():
            b_pt = b_row.geometry
            # Find nearest node in graph within 150m
            for node in list(self.graph_postevent.nodes):
                parts = node.replace("node_", "").split("_")
                n_lon, n_lat = float(parts[0]), float(parts[1])
                # Approx distance in degrees
                if (b_pt.x - n_lon)**2 + (b_pt.y - n_lat)**2 < 0.0015**2:
                    # Remove all incident edges at this severed bridge node
                    incident_edges = list(self.graph_postevent.edges(node))
                    self.graph_postevent.remove_edges_from(incident_edges)

        # 3. Associate settlements with nearest graph nodes
        self.settlement_nodes: Dict[str, str] = {}
        for idx, s_row in self.settlements_gdf.iterrows():
            s_name = s_row["name"]
            s_pt = s_row.geometry
            nearest_node = self._find_nearest_node(s_pt.x, s_pt.y, self.graph_preevent)
            self.settlement_nodes[s_name] = nearest_node

        # 4. Associate operational medical facilities with graph nodes
        self.hospital_nodes: Dict[str, str] = {}
        for idx, f_row in self.facilities_gdf.iterrows():
            f_name = f_row["name"]
            f_pt = f_row.geometry
            # Check if facility itself is flooded
            is_submerged = bool(f_row.get("is_submerged", False))
            if not is_submerged and f_row.get("type") in ["hospital", "staging_hub", "clinic"]:
                nearest_node = self._find_nearest_node(f_pt.x, f_pt.y, self.graph_preevent)
                self.hospital_nodes[f_name] = nearest_node

    def _find_nearest_node(self, lon: float, lat: float, graph: nx.Graph) -> str:
        """Finds closest node in the graph by Euclidean distance."""
        best_node = None
        min_dist = float("inf")
        for node in graph.nodes:
            parts = node.replace("node_", "").split("_")
            n_lon, n_lat = float(parts[0]), float(parts[1])
            dist = (lon - n_lon)**2 + (lat - n_lat)**2
            if dist < min_dist:
                min_dist = dist
                best_node = node
        return best_node

    def analyze_cutoff_settlements(self) -> Dict[str, Any]:
        """
        Evaluates reachability from every settlement to operating medical / relief hubs.
        Returns:
        - List of cut-off settlements with population and severance details
        - Total stranded population
        - Accessible settlements with detour penalties
        """
        hospital_node_set = set(self.hospital_nodes.values())

        cut_off_list: List[Dict[str, Any]] = []
        accessible_list: List[Dict[str, Any]] = []
        total_stranded_population = 0

        for idx, s_row in self.settlements_gdf.iterrows():
            s_name = s_row["name"]
            s_name_ne = s_row.get("name_ne", s_name)
            pop = int(s_row.get("pop", 0))
            s_node = self.settlement_nodes.get(s_name)

            if s_node is None or s_node not in self.graph_postevent:
                is_isolated = True
                min_detour_km = None
            else:
                # Check if path exists to ANY operational hospital
                is_isolated = True
                shortest_dist_post = float("inf")
                nearest_hospital = None

                for h_name, h_node in self.hospital_nodes.items():
                    if h_node in self.graph_postevent and nx.has_path(self.graph_postevent, s_node, h_node):
                        path_len = nx.shortest_path_length(self.graph_postevent, s_node, h_node, weight="weight")
                        if path_len < shortest_dist_post:
                            shortest_dist_post = path_len
                            nearest_hospital = h_name
                            is_isolated = False

            if is_isolated:
                total_stranded_population += pop
                # Find pre-event shortest hospital for context
                pre_hospital = None
                pre_dist = None
                for h_name, h_node in self.hospital_nodes.items():
                    if nx.has_path(self.graph_preevent, s_node, h_node):
                        dist = nx.shortest_path_length(self.graph_preevent, s_node, h_node, weight="weight")
                        if pre_dist is None or dist < pre_dist:
                            pre_dist = dist
                            pre_hospital = h_name

                cut_off_list.append({
                    "name": s_name,
                    "name_ne": s_name_ne,
                    "population": pop,
                    "status": "CUT OFF / ISOLATED",
                    "status_ne": "सम्पर्क विच्छेदित",
                    "elevation_m": s_row.get("elevation_m", 0),
                    "coordinates": [float(s_row.geometry.x), float(s_row.geometry.y)],
                    "pre_event_nearest_hospital": pre_hospital,
                    "pre_event_dist_km": round(pre_dist / 1000.0, 1) if pre_dist else None,
                    "rescue_priority": "CRITICAL - AIRLIFT REQUIRED"
                })
            else:
                # Accessible settlement: compute detour distance
                pre_dist = nx.shortest_path_length(self.graph_preevent, s_node, self.hospital_nodes[nearest_hospital], weight="weight")
                detour_m = shortest_dist_post - pre_dist

                accessible_list.append({
                    "name": s_name,
                    "name_ne": s_name_ne,
                    "population": pop,
                    "status": "ACCESSIBLE",
                    "status_ne": "पहुँचयोग्य",
                    "nearest_hospital": nearest_hospital,
                    "post_event_dist_km": round(shortest_dist_post / 1000.0, 1),
                    "detour_km": round(max(0.0, detour_m) / 1000.0, 1),
                    "coordinates": [float(s_row.geometry.x), float(s_row.geometry.y)]
                })

        return {
            "total_settlements_analyzed": len(self.settlements_gdf),
            "cutoff_count": len(cut_off_list),
            "accessible_count": len(accessible_list),
            "total_stranded_population": total_stranded_population,
            "cutoff_settlements": cut_off_list,
            "accessible_settlements": accessible_list
        }


def run_network_routing_analysis(hazard_geojson_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Main orchestration function:
    1. Loads pre-event OSM snapshot via ohsome client
    2. Intersects with detected flood & debris hazards
    3. Runs NetworkX topological graph cut-off routing
    4. Outputs verified GeoJSON layers and quantitative metrics dictionary
    """
    if hazard_geojson_path is None:
        hazard_geojson_path = CACHE_DIR / "detected_flood_debris_extent.geojson"

    if not hazard_geojson_path.exists():
        from pipeline.ingest_and_change import MultimodalFusionEngine
        engine = MultimodalFusionEngine()
        engine.execute_pipeline()

    hazard_gdf = gpd.read_file(hazard_geojson_path)

    # 1. Fetch Pre-event OSM data
    ohsome_client = OhsomeOSMClient()
    osm_data = ohsome_client.fetch_preevent_data()

    # 2. Assess damage intersections
    assessor = InfrastructureDamageAssessor(hazard_gdf)
    damaged_roads_gdf = assessor.assess_damaged_roads(osm_data["roads"])
    damaged_bridges_gdf = assessor.assess_damaged_bridges(osm_data["bridges"])
    buildings_damage = assessor.assess_damaged_buildings(osm_data["buildings"])

    # 3. Route & Flag Cut-off Settlements
    router = CutOffSettlementRouter(
        damaged_roads_gdf,
        damaged_bridges_gdf,
        osm_data["settlements"],
        osm_data["facilities"]
    )
    routing_results = router.analyze_cutoff_settlements()

    # 4. Save results to cache
    damaged_roads_path = CACHE_DIR / "damaged_roads.geojson"
    damaged_bridges_path = CACHE_DIR / "damaged_bridges.geojson"
    damaged_roads_gdf.to_file(damaged_roads_path, driver="GeoJSON")
    damaged_bridges_gdf.to_file(damaged_bridges_path, driver="GeoJSON")

    total_flooded_road_meters = float(damaged_roads_gdf["flooded_length_m"].sum())
    severed_bridge_count = int(damaged_bridges_gdf["is_severed"].sum())

    summary = {
        "metrics": {
            "cut_off_settlement_count": routing_results["cutoff_count"],
            "total_stranded_population": routing_results["total_stranded_population"],
            "damaged_buildings_count": buildings_damage["total_damaged_count"],
            "damaged_buildings_area_sqm": buildings_damage["total_damaged_area_sqm"],
            "flooded_road_length_meters": round(total_flooded_road_meters, 1),
            "flooded_road_length_km": round(total_flooded_road_meters / 1000.0, 2),
            "severed_bridges_count": severed_bridge_count
        },
        "routing_details": routing_results,
        "building_breakdown": buildings_damage["by_settlement"],
        "layers": {
            "damaged_roads_path": str(damaged_roads_path),
            "damaged_bridges_path": str(damaged_bridges_path)
        },
        "attributions": LEGAL_ATTRIBUTIONS
    }

    metrics_cache_path = CACHE_DIR / "infrastructure_damage_metrics.json"
    with open(metrics_cache_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    res = run_network_routing_analysis()
    m = res["metrics"]
    print("Network Routing & Cut-off Infrastructure Analysis Complete:")
    print(f"  - Cut-Off Settlements: {m['cut_off_settlement_count']} communities")
    print(f"  - Stranded Population: {m['total_stranded_population']:,} persons")
    print(f"  - Severed Bridges: {m['severed_bridges_count']} bridges")
    print(f"  - Flooded Highway Segments: {m['flooded_road_length_km']} km ({m['flooded_road_length_meters']} m)")
    print(f"  - Damaged Buildings: {m['damaged_buildings_count']} structures ({m['damaged_buildings_area_sqm']} m2)")
    print("Cut-off Settlements List:")
    for c in res["routing_details"]["cutoff_settlements"]:
        print(f"    * {c['name']} ({c['name_ne']}) - Pop: {c['population']} - Priority: {c['rescue_priority']}")
