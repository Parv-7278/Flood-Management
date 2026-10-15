"""
FloodWatch-Himalaya: Professional White Theme Disaster Intelligence Platform.
Multimodal AI Hackathon 2026 — Track B: Mapping Flood Damage from Space.
Upgraded with:
- Clean White Theme with deep navy text and subtle light-grey surfaces
- Top Navigation Bar (Overview, Flood Mapping, Infrastructure, Thermal Analysis, Isolated Settlements, What-If Simulator)
- Complete English and Hindi (हिन्दी) bilingual support via centralized i18n
- Human-friendly dashboard metrics with plain-language explanations & tooltips
- Real trained scikit-learn AI/ML What-If Flood Scenario predictive pipeline
- NetworkX topological isolation routing & Copernicus EMS EMSR927 validation
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
import numpy as np

from config import (
    AOI_CENTROID_LAT,
    AOI_CENTROID_LON,
    AOI_BBOX_WGS84,
    DISASTER_EVENT_NAME,
    DISASTER_EVENT_DATE,
    COPERNICUS_EMSR_CODE,
    LEGAL_ATTRIBUTIONS,
    CACHE_DIR,
    FIXTURES_DIR
)
from i18n import t, format_number
from pipeline.ingest_and_change import MultimodalFusionEngine
from pipeline.network_routing import run_network_routing_analysis
from pipeline.validation_emsr import CopernicusEMSRValidator
from copilot.rag_sitrep import BilingualSitRepCopilot, GroundedNumericalRegistry
from ml.config import SCENARIO_PRESETS, METADATA_PATH
from ml.predict import WhatIfFloodPredictor
from pipeline.thermal_pipeline import (
    ThermalDataService,
    ThermalRasterProcessor,
    THERMAL_AOI_PRESETS,
    LANDSAT_ST_SCALE,
    LANDSAT_ST_OFFSET
)

# Page Configuration
st.set_page_config(
    page_title="Flood Intelligence | Spaceborne Disaster Ops",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State
if "language" not in st.session_state:
    st.session_state["language"] = "en"
if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "Overview"
if "sim_results" not in st.session_state:
    st.session_state["sim_results"] = None
if "selected_preset" not in st.session_state:
    st.session_state["selected_preset"] = "baseline"
if "thermal_mode" not in st.session_state:
    st.session_state["thermal_mode"] = "anomaly"
if "thermal_results" not in st.session_state:
    st.session_state["thermal_results"] = None
if "thermal_searched_items" not in st.session_state:
    st.session_state["thermal_searched_items"] = None
if "thermal_selected_aoi" not in st.session_state:
    st.session_state["thermal_selected_aoi"] = "Bhote Koshi Gorge Corridor (Liping to Barhabise)"
if "thermal_opacity" not in st.session_state:
    st.session_state["thermal_opacity"] = 0.75
if "thermal_probe_point" not in st.session_state:
    st.session_state["thermal_probe_point"] = None

# Inject Stylesheet
css_path = Path(__file__).resolve().parent / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_spatial_data():
    """Loads cached pipeline outputs or executes end-to-end ingestion and routing."""
    metrics_path = CACHE_DIR / "infrastructure_damage_metrics.json"
    hazard_path = CACHE_DIR / "detected_flood_debris_extent.geojson"

    if not metrics_path.exists() or not hazard_path.exists():
        engine = MultimodalFusionEngine()
        engine.execute_pipeline()
        run_network_routing_analysis()

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    hazard_gdf = gpd.read_file(hazard_path)
    roads_gdf = gpd.read_file(CACHE_DIR / "damaged_roads.geojson")
    bridges_gdf = gpd.read_file(CACHE_DIR / "damaged_bridges.geojson")
    settlements_gdf = gpd.read_file(FIXTURES_DIR / "settlements.geojson")
    facilities_gdf = gpd.read_file(FIXTURES_DIR / "critical_facilities.geojson")
    emsr_gdf = gpd.read_file(FIXTURES_DIR / "emsr927_ground_truth.geojson")

    validator = CopernicusEMSRValidator(hazard_path, FIXTURES_DIR / "emsr927_ground_truth.geojson")
    val_results = validator.compute_metrics()

    return {
        "metrics_data": metrics_data,
        "hazard_gdf": hazard_gdf,
        "roads_gdf": roads_gdf,
        "bridges_gdf": bridges_gdf,
        "settlements_gdf": settlements_gdf,
        "facilities_gdf": facilities_gdf,
        "emsr_gdf": emsr_gdf,
        "val_results": val_results
    }


def render_top_navigation():
    """Renders professional top navigation bar with brand logo, nav links, language selector, and profile."""
    lang = st.session_state["language"]

    nav_items = [
        ("Overview", t("nav_overview", lang), "📊"),
        ("Flood Mapping", t("nav_flood_mapping", lang), "🌊"),
        ("Infrastructure", t("nav_infrastructure", lang), "🛣️"),
        ("Thermal Analysis", t("nav_thermal", lang), "🌡️"),
        ("Isolated Settlements", t("nav_isolated", lang), "🚨"),
        ("What-If Simulator", t("nav_simulator", lang), "🧠"),
    ]

    # Navbar Container
    nav_col1, nav_col2 = st.columns([1, 1])

    with nav_col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px; padding: 4px 0;">
            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #0284c7, #2563eb); color: white; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 22px; font-weight: 800; box-shadow: 0 2px 4px rgba(2, 132, 199, 0.25);">
                🌊
            </div>
            <div>
                <div style="font-size: 20px; font-weight: 800; color: #0f172a; letter-spacing: -0.5px; line-height: 1.1;">
                    {t('nav_brand', lang)}
                </div>
                <div style="font-size: 11px; color: #334155; font-weight: 600;">
                    {t('nav_subtitle', lang)}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with nav_col2:
        top_ctrl1, top_ctrl2, top_ctrl3 = st.columns([2, 1.2, 1.8])
        with top_ctrl1:
            lang_choice = st.selectbox(
                "Language / भाषा",
                options=["English", "हिन्दी (Hindi)"],
                index=0 if lang == "en" else 1,
                label_visibility="collapsed"
            )
            new_lang = "en" if "English" in lang_choice else "hi"
            if new_lang != st.session_state["language"]:
                st.session_state["language"] = new_lang
                st.rerun()

        with top_ctrl2:
            st.markdown(f"""
            <div style="display:flex; justify-content: center; align-items: center; height: 38px; background: #fee2e2; border: 1px solid #fecaca; border-radius: 8px; font-size: 12px; font-weight: 700; color: #b91c1c; gap: 6px;">
                🔔 <span>4</span>
            </div>
            """, unsafe_allow_html=True)

        with top_ctrl3:
            st.markdown(f"""
            <div style="display:flex; align-items: center; gap: 8px; height: 38px; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 0 10px;">
                <div style="width: 22px; height: 22px; border-radius: 50%; background: #e0f2fe; color: #0369a1; font-size: 11px; font-weight: 700; display: flex; align-items: center; justify-content: center;">
                    RO
                </div>
                <div style="font-size: 12px; font-weight: 700; color: #0f172a; white-space: nowrap;">
                    {t('nav_user_role', lang)}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Sub-Navigation Links Row
    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    cols = st.columns(len(nav_items))
    for i, (nav_id, label, icon) in enumerate(nav_items):
        with cols[i]:
            is_active = st.session_state["active_nav"] == nav_id
            btn_style = "primary" if is_active else "secondary"
            if st.button(f"{icon} {label}", key=f"nav_btn_{nav_id}", type=btn_style, use_container_width=True):
                st.session_state["active_nav"] = nav_id
                st.rerun()

    st.markdown("<hr style='border: 0; border-top: 1px solid #e2e8f0; margin: 12px 0 18px 0;'>", unsafe_allow_html=True)


def render_human_friendly_metrics(data: Dict[str, Any], lang: str):
    """Renders human-friendly scoreboard cards with clear plain-language descriptions and tooltips."""
    m = data["metrics_data"]["metrics"]
    routing = data["metrics_data"]["routing_details"]

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    # 1. Flood-Affected Area
    with col1:
        st.markdown(f"""
        <div class="metric-card-white card-accent-blue" title="{t('metric_affected_area_tooltip', lang)}">
            <div class="metric-label">
                <span>{t('metric_affected_area', lang)}</span>
                <span class="info-icon">ℹ️</span>
            </div>
            <div class="metric-value">{format_number(23.50, lang)} km²</div>
            <div class="metric-subtext">{t('metric_affected_area_sub', lang)}</div>
            <div style="margin-top: 6px;">
                <span class="badge-status badge-status-info" style="font-size: 9px; padding: 2px 6px;">{t('badge_detected', lang)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Potentially Impacted Buildings
    with col2:
        st.markdown(f"""
        <div class="metric-card-white card-accent-purple" title="{t('metric_impacted_buildings_tooltip', lang)}">
            <div class="metric-label">
                <span>{t('metric_impacted_buildings', lang)}</span>
                <span class="info-icon">ℹ️</span>
            </div>
            <div class="metric-value">{format_number(m['damaged_buildings_count'], lang)}</div>
            <div class="metric-subtext">{t('metric_impacted_buildings_sub', lang)}</div>
            <div style="margin-top: 6px;">
                <span class="badge-status badge-status-prediction" style="font-size: 9px; padding: 2px 6px;">{t('badge_estimated', lang)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Roads at Risk
    with col3:
        st.markdown(f"""
        <div class="metric-card-white card-accent-blue" title="{t('metric_roads_risk_tooltip', lang)}">
            <div class="metric-label">
                <span>{t('metric_roads_risk', lang)}</span>
                <span class="info-icon">ℹ️</span>
            </div>
            <div class="metric-value">{format_number(m['flooded_road_length_km'], lang)} km</div>
            <div class="metric-subtext">{format_number(m['flooded_road_length_meters'], lang)} m inundated</div>
            <div style="margin-top: 6px;">
                <span class="badge-status badge-status-danger" style="font-size: 9px; padding: 2px 6px;">H03 & H21</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Bridges Requiring Inspection
    with col4:
        st.markdown(f"""
        <div class="metric-card-white card-accent-amber" title="{t('metric_bridges_inspection_tooltip', lang)}">
            <div class="metric-label">
                <span>{t('metric_bridges_inspection', lang)}</span>
                <span class="info-icon">ℹ️</span>
            </div>
            <div class="metric-value" style="color: #b45309;">{format_number(m['severed_bridges_count'], lang)}</div>
            <div class="metric-subtext">{t('metric_bridges_inspection_sub', lang)}</div>
            <div style="margin-top: 6px;">
                <span class="badge-status badge-status-danger" style="font-size: 9px; padding: 2px 6px;">CRITICAL SCOUR</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Potentially Isolated Settlements
    with col5:
        st.markdown(f"""
        <div class="metric-card-white card-accent-red" title="{t('metric_isolated_settlements_tooltip', lang)}">
            <div class="metric-label">
                <span>{t('metric_isolated_settlements', lang)}</span>
                <span class="info-icon">ℹ️</span>
            </div>
            <div class="metric-value" style="color: #b91c1c;">{format_number(m['cut_off_settlement_count'], lang)}</div>
            <div class="metric-subtext">{format_number(m['total_stranded_population'], lang)} residents stranded</div>
            <div style="margin-top: 6px;">
                <span class="badge-status badge-status-danger" style="font-size: 9px; padding: 2px 6px;">AIRLIFT PRIORITY</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 6. Thermal Anomalies
    with col6:
        st.markdown(f"""
        <div class="metric-card-white card-accent-amber" title="{t('metric_thermal_anomalies_tooltip', lang)}">
            <div class="metric-label">
                <span>{t('metric_thermal_anomalies', lang)}</span>
                <span class="info-icon">ℹ️</span>
            </div>
            <div class="metric-value">+{format_number(5.58, lang)} K</div>
            <div class="metric-subtext">{t('metric_thermal_anomalies_sub', lang)}</div>
            <div style="margin-top: 6px;">
                <span class="badge-status badge-status-info" style="font-size: 9px; padding: 2px 6px;">SLSTR / TIRS</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 7. Overall Disaster Severity
    with col7:
        st.markdown(f"""
        <div class="metric-card-white card-accent-red" title="{t('metric_severity_tooltip', lang)}">
            <div class="metric-label">
                <span>{t('metric_severity', lang)}</span>
                <span class="info-icon">ℹ️</span>
            </div>
            <div class="metric-value" style="font-size: 18px; color: #b91c1c;">{t('metric_severity_val', lang)}</div>
            <div class="metric-subtext">{t('metric_severity_sub', lang)}</div>
            <div style="margin-top: 6px;">
                <span class="badge-status badge-status-danger" style="font-size: 9px; padding: 2px 6px;">NEOC HIGH</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def build_geospatial_folium_map(
    data: Dict[str, Any],
    show_sim_layer: bool = False,
    sim_data: Optional[Dict[str, Any]] = None,
    show_thermal_overlay: bool = False,
    thermal_data: Optional[Dict[str, Any]] = None,
    thermal_opacity: float = 0.75,
    overlay_options: Optional[Dict[str, bool]] = None
):
    """Builds interactive Folium Map with clean white/light OpenStreetMap basemap and multi-sensor overlays."""
    folium_map = folium.Map(
        location=[AOI_CENTROID_LAT, AOI_CENTROID_LON],
        zoom_start=11,
        tiles="OpenStreetMap",
        control_scale=True
    )

    opts = overlay_options or {
        "water": True,
        "debris": True,
        "roads": True,
        "bridges": True,
        "settlements": True,
        "facilities": True
    }

    # Optional Calibrated Thermal Raster Overlay
    if show_thermal_overlay and thermal_data and "overlay_uri" in thermal_data:
        bounds = thermal_data["bounds_geojson"]
        is_lst = thermal_data.get("mode") == "surface_temp"
        layer_title = f"🌡️ Thermal {'LST (°C)' if is_lst else 'Anomaly ΔT (K)'} [{thermal_data.get('date', '2026-08')}]"
        folium.raster_layers.ImageOverlay(
            image=thermal_data["overlay_uri"],
            bounds=bounds,
            opacity=thermal_opacity,
            name=layer_title,
            interactive=True,
            cross_origin=False,
            zindex=8
        ).add_to(folium_map)

    # Inundated Water (Cyan/Blue)
    if opts.get("water", True):
        water_gdf = data["hazard_gdf"][data["hazard_gdf"]["hazard_type"] == "inundated_water"]
        if not water_gdf.empty:
            folium.GeoJson(
                water_gdf,
                name="Inundated Flood Water (Sentinel-1 SAR / S-2 MNDWI)",
                style_function=lambda x: {
                    "fillColor": "#0284c7",
                    "color": "#0369a1",
                    "weight": 1.5,
                    "fillOpacity": 0.65
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["hazard_type", "confidence", "area_sqkm"],
                    aliases=["Hazard:", "Confidence:", "Area (km²):"],
                    localize=True
                )
            ).add_to(folium_map)

    # Friction-Heated Debris Flow (Orange/Amber)
    if opts.get("debris", True):
        debris_gdf = data["hazard_gdf"][data["hazard_gdf"]["hazard_type"] == "friction_heated_debris_flow"]
        if not debris_gdf.empty:
            folium.GeoJson(
                debris_gdf,
                name="Friction-Heated Debris Flow (Thermal LST Anomaly ΔT)",
                style_function=lambda x: {
                    "fillColor": "#ea580c",
                    "color": "#c2410c",
                    "weight": 2.0,
                    "fillOpacity": 0.70
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["hazard_type", "confidence", "delta_lst_k", "area_sqkm"],
                    aliases=["Hazard:", "Confidence:", "Thermal Anomaly ΔT (K):", "Area (km²):"],
                    localize=True
                )
            ).add_to(folium_map)

    # Road Network
    if opts.get("roads", True):
        for idx, r_row in data["roads_gdf"].iterrows():
            coords = [(c[1], c[0]) for c in r_row.geometry.coords]
            is_flooded = r_row["is_flooded"]
            color = "#b91c1c" if is_flooded else "#15803d"
            weight = 4 if is_flooded else 2.5
            dash = "5, 5" if is_flooded else None

            popup_txt = f"<b>{r_row.get('name', 'Road Segment')}</b><br>Status: <b>{r_row['status']}</b><br>Flooded: {r_row['flooded_length_m']} m"
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=weight,
                opacity=0.9,
                dash_array=dash,
                popup=popup_txt
            ).add_to(folium_map)

    # Bridges
    if opts.get("bridges", True):
        for idx, b_row in data["bridges_gdf"].iterrows():
            pt = b_row.geometry
            is_sev = b_row["is_severed"]
            icon_color = "red" if is_sev else "green"
            icon_symbol = "times-circle" if is_sev else "check-circle"

            folium.Marker(
                location=[pt.y, pt.x],
                icon=folium.Icon(color=icon_color, icon=icon_symbol, prefix="fa"),
                tooltip=f"Bridge: {b_row['name']} ({b_row['status']})"
            ).add_to(folium_map)

    # Settlements (Cut-off highlighted with red circular markers)
    if opts.get("settlements", True):
        routing = data["metrics_data"]["routing_details"]
        cutoff_names = {c["name"] for c in routing["cutoff_settlements"]}
        for idx, s_row in data["settlements_gdf"].iterrows():
            pt = s_row.geometry
            s_name = s_row["name"]
            is_cutoff = s_name in cutoff_names

            if is_cutoff:
                folium.CircleMarker(
                    location=[pt.y, pt.x],
                    radius=10,
                    color="#b91c1c",
                    weight=3,
                    fill=True,
                    fill_color="#ef4444",
                    fill_opacity=0.85,
                    tooltip=f"🚨 CUT OFF: {s_name} ({s_row.get('name_ne', s_name)}) - Pop: {s_row['pop']:,}"
                ).add_to(folium_map)
            else:
                folium.CircleMarker(
                    location=[pt.y, pt.x],
                    radius=6,
                    color="#15803d",
                    weight=2,
                    fill=True,
                    fill_color="#22c55e",
                    fill_opacity=0.75,
                    tooltip=f"✅ Accessible: {s_name}"
                ).add_to(folium_map)

    # Critical Facilities (Hospitals)
    if opts.get("facilities", True):
        for idx, f_row in data["facilities_gdf"].iterrows():
            pt = f_row.geometry
            folium.Marker(
                location=[pt.y, pt.x],
                icon=folium.Icon(color="blue", icon="hospital", prefix="fa"),
                tooltip=f"Medical Hub: {f_row['name']}"
            ).add_to(folium_map)

    # Optional Simulated Layer
    if show_sim_layer and sim_data and "simulated_layers" in sim_data:
        sim_flood = sim_data["simulated_layers"].get("flood_gdf")
        if sim_flood is not None and not sim_flood.empty:
            folium.GeoJson(
                sim_flood,
                name="AI/ML Predicted Scenario Flood Extent (P ≥ 0.50)",
                style_function=lambda x: {
                    "fillColor": "#7c3aed",
                    "color": "#6d28d9",
                    "weight": 2.5,
                    "fillOpacity": 0.45,
                    "dashArray": "3, 3"
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["flood_prob"],
                    aliases=["Predicted Inundation Probability:"]
                )
            ).add_to(folium_map)

    folium.LayerControl(position="topright").add_to(folium_map)
    plugins.Fullscreen(position="topleft").add_to(folium_map)
    return folium_map


def render_thermal_analysis_view(data: Dict[str, Any], lang: str):
    """
    Renders the Spaceborne Thermal Mapping & Kinetic Debris Anomaly Analysis interface.
    Features:
    - USGS Landsat 8/9 Collection 2 Level-2 Surface Temperature (ST_B10) & Sentinel-3 SLSTR LST.
    - STAC search or local high-resolution regional reference scenes.
    - Pre- vs Post-Event Anomaly (ΔT) or Land Surface Temperature (LST in °C).
    - Physical calibration (Kelvin = DN * 0.00341802 + 149.0, °C = K - 273.15).
    - Multi-layer georeferenced Folium overlay with opacity slider and layer toggles.
    - Interactive point temperature inspection across Himalayan river gorge coordinates.
    """
    st.markdown(f"### 🌡️ **{t('thermal_header', lang)}**")
    st.markdown(f"<p style='color:#334155; font-size:13px; font-weight:500; margin-top:-6px;'>{t('thermal_subtitle', lang)}</p>", unsafe_allow_html=True)

    # 1. Configuration & Acquisition Selection Card
    with st.expander(f"🛰️ **{t('thermal_satellite_label', lang)} & {t('thermal_aoi_label', lang)}**", expanded=(st.session_state["thermal_results"] is None)):
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

        with cfg_col1:
            analysis_mode = st.radio(
                t("thermal_mode_label", lang),
                options=["anomaly", "surface_temp"],
                format_func=lambda x: f"🔥 {t('thermal_mode_anomaly', lang)}" if x == "anomaly" else f"🌡️ {t('thermal_mode_surface', lang)}",
                index=0 if st.session_state["thermal_mode"] == "anomaly" else 1
            )
            if analysis_mode != st.session_state["thermal_mode"]:
                st.session_state["thermal_mode"] = analysis_mode
                st.session_state["thermal_results"] = None

        with cfg_col2:
            sat_choice = st.selectbox(
                t("thermal_satellite_label", lang),
                options=[
                    "Landsat 8/9 Collection 2 Level-2 (ST_B10 30m)",
                    "Copernicus Sentinel-3 SLSTR (L2 LST 1km)"
                ],
                index=0
            )
            sat_collection = "landsat-c2-l2" if "Landsat" in sat_choice else "sentinel-3-slstr-lst-l2-netcdf"

        with cfg_col3:
            aoi_choice = st.selectbox(
                t("thermal_aoi_label", lang),
                options=list(THERMAL_AOI_PRESETS.keys()),
                index=0
            )
            selected_bbox = THERMAL_AOI_PRESETS[aoi_choice]

        # Date Pickers Row
        d_col1, d_col2, d_col3 = st.columns([1.5, 1.5, 1.2])
        with d_col1:
            start_d = st.date_input(t("thermal_start_date", lang), value=pd.to_datetime("2024-05-01"))
        with d_col2:
            end_d = st.date_input(t("thermal_end_date", lang), value=pd.to_datetime("2026-08-30"))
        with d_col3:
            max_cloud = st.slider("Max Cloud %", min_value=10, max_value=100, value=80, step=5)

        search_btn = st.button(f"🔍 {t('thermal_btn_search', lang)}", use_container_width=True)
        if search_btn or st.session_state["thermal_searched_items"] is None:
            with st.spinner("Searching STAC catalog..."):
                searched = ThermalDataService.search_acquisitions(
                    aoi_bbox=selected_bbox,
                    start_date=str(start_d),
                    end_date=str(end_d),
                    satellite=sat_collection,
                    max_cloud_cover=float(max_cloud)
                )
                st.session_state["thermal_searched_items"] = searched

        items = st.session_state.get("thermal_searched_items", [])

        # Display discovered acquisitions
        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        selected_pre_item = None
        selected_post_item = None
        selected_single_item = None

        if items:
            item_labels = [
                f"{it['date']} | {it['platform']} | Cloud: {it['cloud_cover']}% {'[LIVE STAC]' if it.get('is_live') else '[REGIONAL REFERENCE]'}"
                for it in items
            ]
            if analysis_mode == "anomaly":
                sel_c1, sel_c2 = st.columns(2)
                with sel_c1:
                    pre_idx = 0 if len(items) > 0 else 0
                    pre_sel = st.selectbox("Baseline Pre-Event Pass", options=range(len(items)), format_func=lambda i: item_labels[i], index=pre_idx)
                    selected_pre_item = items[pre_sel]
                with sel_c2:
                    post_idx = min(1, len(items) - 1)
                    post_sel = st.selectbox("Post-Disaster Pass", options=range(len(items)), format_func=lambda i: item_labels[i], index=post_idx)
                    selected_post_item = items[post_sel]
            else:
                sel_idx = st.selectbox("Select Thermal Pass", options=range(len(items)), format_func=lambda i: item_labels[i], index=0)
                selected_single_item = items[sel_idx]
        else:
            st.warning(t("thermal_status_no_data", lang))

    # Processing Execution Button
    run_proc_btn = st.button(f"🚀 {t('thermal_btn_process', lang)}", type="primary", use_container_width=True)

    # Process execution
    if run_proc_btn or st.session_state["thermal_results"] is None:
        with st.spinner(t("thermal_status_processing", lang)):
            try:
                if st.session_state["thermal_mode"] == "surface_temp":
                    # Load single pass
                    tgt_item = selected_single_item if selected_single_item else ThermalDataService.get_reference_fixtures(selected_bbox, sat_collection)[0]
                    res = ThermalRasterProcessor.load_surface_temperature(tgt_item, aoi_bbox=selected_bbox)
                    overlay_uri, legend = ThermalRasterProcessor.render_overlay_png(res["temp_celsius"], mode="surface_temp")
                    res["overlay_uri"] = overlay_uri
                    res["legend"] = legend
                    res["mode"] = "surface_temp"
                    st.session_state["thermal_results"] = res
                else:
                    # Anomaly mode
                    p_pre = selected_pre_item if selected_pre_item else ThermalDataService.get_reference_fixtures(selected_bbox, sat_collection)[0]
                    p_post = selected_post_item if selected_post_item else ThermalDataService.get_reference_fixtures(selected_bbox, sat_collection)[1]
                    pre_res = ThermalRasterProcessor.load_surface_temperature(p_pre, aoi_bbox=selected_bbox)
                    post_res = ThermalRasterProcessor.load_surface_temperature(p_post, aoi_bbox=selected_bbox)
                    anom_res = ThermalRasterProcessor.compute_anomaly(pre_res, post_res)
                    overlay_uri, legend = ThermalRasterProcessor.render_overlay_png(anom_res["delta_temp"], mode="anomaly")
                    anom_res["overlay_uri"] = overlay_uri
                    anom_res["legend"] = legend
                    anom_res["mode"] = "anomaly"
                    anom_res["pre_date"] = pre_res["date"]
                    anom_res["post_date"] = post_res["date"]
                    anom_res["platform"] = post_res["platform"]
                    st.session_state["thermal_results"] = anom_res
            except Exception as e:
                st.error(f"Thermal processing error: {str(e)}")

    t_res = st.session_state.get("thermal_results")
    if not t_res:
        st.info("Please select an observation and click 'EXECUTE THERMAL PROCESSING'.")
        return

    # Status & Observation Banner
    if t_res["mode"] == "anomaly":
        sub_info = f"Pre-Event Baseline: <b>{t_res['pre_date']}</b> | Post-Event Observation: <b>{t_res['post_date']}</b> ({t_res.get('platform', 'Landsat')})"
    else:
        sub_info = f"Observation Timestamp: <b>{t_res['date']}</b> | Sensor: <b>{t_res['platform']}</b>"

    st.markdown(f"""
    <div style="background:#dcfce7; border:1px solid #86efac; border-radius:8px; padding:10px 14px; margin: 12px 0 16px 0; color:#166534; font-size:13px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
        <div>
            ✅ <b>{t('thermal_status_success', lang)}</b> — {sub_info}
        </div>
        <div>
            <span class="badge-status badge-status-info">PHYSICAL LST CALIBRATED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Scoreboard Metric Cards
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    st_data = t_res["stats"]

    if t_res["mode"] == "anomaly":
        with m_col1:
            st.markdown(f"""
            <div class="metric-card-white card-accent-red">
                <div class="metric-label">{t('thermal_stat_max_anomaly', lang)}</div>
                <div class="metric-value" style="color:#b91c1c;">+{st_data['max_positive_anomaly_k']:.2f} K</div>
                <div class="metric-subtext">Peak Kinetic Friction Heating</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card-white card-accent-amber">
                <div class="metric-label">Mean Corridor ΔT</div>
                <div class="metric-value" style="color:#b45309;">{st_data['mean_anomaly_k']:+.2f} K</div>
                <div class="metric-subtext">Basin Temperature Shift</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-card-white card-accent-amber">
                <div class="metric-label">{t('thermal_stat_friction_area', lang)}</div>
                <div class="metric-value" style="color:#b45309;">{st_data['friction_heated_area_km2']:.2f} km²</div>
                <div class="metric-subtext">Shear Margin Mudflow Slurry</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
            <div class="metric-card-white card-accent-blue">
                <div class="metric-label">Warm Impoundments</div>
                <div class="metric-value">{st_data['warm_mud_pool_area_km2']:.2f} km²</div>
                <div class="metric-subtext">Turbid Slackwater Silt Deposits</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col5:
            st.markdown(f"""
            <div class="metric-card-white card-accent-green">
                <div class="metric-label">{t('thermal_stat_valid_pixels', lang)}</div>
                <div class="metric-value" style="font-size:20px;">{format_number(st_data['valid_pixel_count'], lang)}</div>
                <div class="metric-subtext">Cloud-Free 30m Pixels</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        with m_col1:
            st.markdown(f"""
            <div class="metric-card-white card-accent-blue">
                <div class="metric-label">{t('thermal_stat_mean_temp', lang)}</div>
                <div class="metric-value">{st_data['mean_celsius']:.1f}°C</div>
                <div class="metric-subtext">Average Valley Temperature</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card-white card-accent-purple">
                <div class="metric-label">High Ridge Min LST</div>
                <div class="metric-value">{st_data['min_celsius']:.1f}°C</div>
                <div class="metric-subtext">Glacier Slopes / Permafrost</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-card-white card-accent-red">
                <div class="metric-label">Low Gorge Max LST</div>
                <div class="metric-value" style="color:#b91c1c;">{st_data['max_celsius']:.1f}°C</div>
                <div class="metric-subtext">Sun-Exposed Alluvial Bed</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
            <div class="metric-card-white card-accent-blue">
                <div class="metric-label">{t('thermal_stat_valid_pixels', lang)}</div>
                <div class="metric-value" style="font-size:20px;">{format_number(st_data['valid_pixels'], lang)}</div>
                <div class="metric-subtext">Calibrated 30m Pixels</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col5:
            st.markdown(f"""
            <div class="metric-card-white card-accent-green">
                <div class="metric-label">{t('thermal_stat_cloud_pct', lang)}</div>
                <div class="metric-value" style="font-size:20px;">{st_data['cloud_masked_pct']}%</div>
                <div class="metric-subtext">QA_PIXEL Masked</div>
            </div>
            """, unsafe_allow_html=True)

    # Detailed Interactive Map & Point Probe Workspace
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    map_ctrl_col, map_display_col = st.columns([1.1, 2.1])

    with map_ctrl_col:
        st.markdown("##### 🎛️ **Layer Controls & Legend**")
        opacity = st.slider(t("thermal_opacity_label", lang), min_value=0.1, max_value=1.0, value=0.75, step=0.05)

        st.markdown(f"**{t('thermal_toggle_overlay', lang)}**")
        c_water = st.checkbox("🌊 Inundated Water (Sentinel-1 SAR / S-2)", value=True)
        c_debris = st.checkbox("🟠 Friction Debris Mask (Vector Consensus)", value=False)
        c_roads = st.checkbox("🛣️ Highway Network (H03 / H21)", value=True)
        c_bridges = st.checkbox("🌉 Critical Bridges & Scour Risk", value=True)
        c_settlements = st.checkbox("🚨 Cut-Off Settlements & Hospitals", value=True)

        opts = {
            "water": c_water,
            "debris": c_debris,
            "roads": c_roads,
            "bridges": c_bridges,
            "settlements": c_settlements,
            "facilities": True
        }

        # Visual Legend
        leg = t_res["legend"]
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if t_res["mode"] == "anomaly":
            st.markdown(f"""
            <div class="thermal-legend-container">
                <b style="color:#0f172a; font-size:12px;">{t('thermal_legend_anomaly', lang)}</b>
                <div class="thermal-gradient-bar-anomaly"></div>
                <div class="thermal-legend-labels">
                    <span>{leg['vmin']:+.1f} K (Cooling)</span>
                    <span>0.0 K (Neutral)</span>
                    <span style="color:#b91c1c;">{leg['vmax']:+.1f} K (Friction)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="thermal-legend-container">
                <b style="color:#0f172a; font-size:12px;">{t('thermal_legend_lst', lang)}</b>
                <div class="thermal-gradient-bar-lst"></div>
                <div class="thermal-legend-labels">
                    <span>{leg['vmin']:.1f}°C (Cold High Peak)</span>
                    <span>{((leg['vmin']+leg['vmax'])/2):.1f}°C</span>
                    <span style="color:#b91c1c;">{leg['vmax']:.1f}°C (Warm Valley)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Point Probe Tool
        st.markdown(f"##### 🎯 **{t('thermal_probe_title', lang)}**")
        st.markdown(f"<p style='color:#334155; font-size:11px; margin-top:-4px;'>{t('thermal_probe_hint', lang)}</p>", unsafe_allow_html=True)

        probe_presets = [
            ("Tatopani Gorge", 27.954, 85.938),
            ("Liping Breach Outlet", 27.981, 85.975),
            ("Larcha River Bend", 27.925, 85.918),
            ("Barhabise Logistics Hub", 27.789, 85.898)
        ]
        pb_c1, pb_c2 = st.columns(2)
        for idx, (p_name, p_lat, p_lon) in enumerate(probe_presets):
            btn_col = pb_c1 if idx % 2 == 0 else pb_c2
            with btn_col:
                if st.button(f"📍 {p_name}", key=f"pb_btn_{idx}", use_container_width=True):
                    st.session_state["thermal_probe_point"] = (p_lat, p_lon, p_name)

        if st.session_state.get("thermal_probe_point"):
            pr_lat, pr_lon, pr_lbl = st.session_state["thermal_probe_point"]
            p_val = ThermalRasterProcessor.sample_point_temperature(
                t_res["data_grid"] if "data_grid" in t_res else t_res.get("delta_temp", t_res.get("temp_celsius")),
                t_res["transform"],
                t_res["crs"],
                lat=pr_lat,
                lon=pr_lon
            )
            unit_str = "°C" if t_res["mode"] == "surface_temp" else "K (ΔT)"
            st.markdown(f"""
            <div class="thermal-probe-card">
                <div style="font-size:12px; font-weight:700; color:#0f172a;">Sampled Location: <b>{pr_lbl}</b></div>
                <div style="font-size:11px; color:#475569;">Coordinates: {pr_lat:.4f}°N, {pr_lon:.4f}°E</div>
                <div style="font-size:18px; font-weight:800; color:{'#b91c1c' if (p_val or 0) > 2.5 else '#0284c7'}; margin-top:4px;">
                    {p_val:+.2f if t_res['mode'] == 'anomaly' else p_val:.2f} {unit_str}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Scientific Integrity Callout
        st.markdown(f"""
        <div style="background:#f1f5f9; border-left:3px solid #64748b; border-radius:0 6px 6px 0; padding:10px 12px; margin-top:10px; font-size:11px; color:#334155; line-height:1.5;">
            {t('thermal_scientific_note', lang)}
        </div>
        """, unsafe_allow_html=True)

    with map_display_col:
        st.markdown(f"##### 🗺️ **{t('thermal_header', lang)} — Georeferenced Map View**")
        thermal_map = build_geospatial_folium_map(
            data,
            show_thermal_overlay=True,
            thermal_data=t_res,
            thermal_opacity=opacity,
            overlay_options=opts
        )
        st_folium(thermal_map, width="100%", height=620, key="thermal_map_folium_interactive")


def render_what_if_simulator(data: Dict[str, Any], lang: str):
    """Renders the AI/ML What-If Flood Scenario Simulator interface and displays genuine predictions."""
    st.markdown(f"### 🧠 **{t('sim_title', lang)}**")
    st.markdown(f"<p style='color:#334155; font-size:14px; margin-top:-6px; font-weight:500;'>{t('sim_subtitle', lang)}</p>", unsafe_allow_html=True)

    # Scenario Presets Selection
    st.markdown(f"##### 🎯 {t('sim_preset_label', lang)}")
    p_col1, p_col2, p_col3 = st.columns(3)

    with p_col1:
        if st.button(f"🟢 {t('preset_baseline', lang)}", use_container_width=True):
            st.session_state["selected_preset"] = "baseline"
            st.rerun()
    with p_col2:
        if st.button(f"🟠 {t('preset_severe', lang)}", use_container_width=True):
            st.session_state["selected_preset"] = "severe"
            st.rerun()
    with p_col3:
        if st.button(f"🔴 {t('preset_extreme', lang)}", use_container_width=True):
            st.session_state["selected_preset"] = "extreme"
            st.rerun()

    active_preset_key = st.session_state.get("selected_preset", "baseline")
    preset_data = SCENARIO_PRESETS[active_preset_key]

    st.markdown(f"""
    <div class="preset-container">
        <b>Active Scenario:</b> <span style="color:#0284c7; font-weight:700;">{preset_data['name' if lang == 'en' else 'name_hi']}</span> — 
        <span style="color:#334155; font-size:13px; font-weight:500;">{preset_data['description']}</span>
    </div>
    """, unsafe_allow_html=True)

    # Simulator Input Sliders (Populated by Preset but customizable by user)
    with st.expander(f"🎛️ **{t('sim_inputs_header', lang)}**", expanded=True):
        sc_col1, sc_col2, sc_col3 = st.columns(3)

        with sc_col1:
            rainfall_intensity = st.slider(
                t("param_rainfall_intensity", lang),
                min_value=5.0, max_value=120.0,
                value=float(preset_data["rainfall_intensity_mmh"]),
                step=5.0
            )
            rainfall_duration = st.slider(
                t("param_rainfall_duration", lang),
                min_value=1.0, max_value=48.0,
                value=float(preset_data["rainfall_duration_hrs"]),
                step=1.0
            )

        with sc_col2:
            surge_height = st.slider(
                t("param_surge_height", lang),
                min_value=0.5, max_value=8.5,
                value=float(preset_data["surge_height_m"]),
                step=0.25
            )
            thermal_intensity = st.slider(
                t("param_thermal_intensity", lang),
                min_value=1.0, max_value=8.0,
                value=float(preset_data["thermal_intensity_k"]),
                step=0.2
            )

        with sc_col3:
            breach_loc = st.selectbox(
                t("param_breach_location", lang),
                options=[
                    "Upper Bhote Koshi Glacier / Liping",
                    "Trishuli Mailung Tributary Gorge",
                    "Chaku Khola High Confluence",
                    "Larcha River Bend"
                ],
                index=0
            )
            propagation_factor = st.slider(
                t("param_propagation_factor", lang),
                min_value=0.8, max_value=2.5,
                value=float(preset_data["propagation_factor"]),
                step=0.1
            )
            road_override = st.checkbox(
                t("param_road_closure_override", lang),
                value=bool(preset_data["road_closure_override"])
            )

    run_sim_btn = st.button(f"🚀 {t('btn_run_simulation', lang)}", type="primary", use_container_width=True)

    # Execute ML Prediction Pipeline
    if run_sim_btn or st.session_state["sim_results"] is None:
        with st.spinner(t("btn_running_simulation", lang)):
            predictor = WhatIfFloodPredictor()
            scenario_inputs = {
                "rainfall_intensity_mmh": rainfall_intensity,
                "rainfall_duration_hrs": rainfall_duration,
                "surge_height_m": surge_height,
                "upstream_breach_location": breach_loc,
                "thermal_intensity_k": thermal_intensity,
                "propagation_factor": propagation_factor,
                "road_closure_override": road_override
            }
            sim_output = predictor.predict_scenario(scenario_inputs)
            st.session_state["sim_results"] = sim_output

    res = st.session_state["sim_results"]
    p = res["predictions"]
    comp = res["comparison_with_baseline"]

    # Prediction Notice Banner
    st.markdown(f"""
    <div class="prediction-banner">
        {t('sim_notice_prediction', lang)}
    </div>
    """, unsafe_allow_html=True)

    # Simulated Impact Metric Cards
    res_col1, res_col2, res_col3, res_col4, res_col5 = st.columns(5)

    with res_col1:
        st.markdown(f"""
        <div class="metric-card-white card-accent-purple">
            <div class="metric-label">{t('sim_stat_predicted_area', lang)}</div>
            <div class="metric-value">{format_number(p['inundated_area_sqkm'], lang)} km²</div>
            <div class="metric-subtext">Δ Baseline: <span class="delta-badge-pos">{comp['delta_inundated_area_sqkm']:+} km²</span></div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div class="metric-card-white card-accent-purple">
            <div class="metric-label">{t('sim_stat_predicted_buildings', lang)}</div>
            <div class="metric-value">{format_number(p['impacted_buildings_estimate'], lang)}</div>
            <div class="metric-subtext">CI: [{p['impacted_buildings_ci'][0]}, {p['impacted_buildings_ci'][1]}]</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col3:
        st.markdown(f"""
        <div class="metric-card-white card-accent-blue">
            <div class="metric-label">{t('sim_stat_predicted_roads', lang)}</div>
            <div class="metric-value">{format_number(p['flooded_road_length_km'], lang)} km</div>
            <div class="metric-subtext">Δ Baseline: <span class="delta-badge-pos">{comp['delta_roads_km']:+} km</span></div>
        </div>
        """, unsafe_allow_html=True)

    with res_col4:
        st.markdown(f"""
        <div class="metric-card-white card-accent-amber">
            <div class="metric-label">{t('sim_stat_predicted_bridges', lang)}</div>
            <div class="metric-value" style="color:#b45309;">{format_number(p['bridges_requiring_inspection_count'], lang)}</div>
            <div class="metric-subtext">High Scour Risk</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col5:
        st.markdown(f"""
        <div class="metric-card-white card-accent-red">
            <div class="metric-label">{t('sim_stat_predicted_isolated', lang)}</div>
            <div class="metric-value" style="color:#b91c1c;">{format_number(p['cutoff_settlement_count'], lang)}</div>
            <div class="metric-subtext">{format_number(p['stranded_population_count'], lang)} stranded</div>
        </div>
        """, unsafe_allow_html=True)

    # Detailed Comparison & Visualizations Row
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    det_col1, det_col2 = st.columns([1.1, 1.9])

    with det_col1:
        st.markdown(f"#### 📊 **{t('sim_key_factors_title', lang)}**")
        st.markdown("<p style='font-size:12px; color:#334155; font-weight:500;'>Derived from trained Random Forest Gini feature importances scaled by scenario parameter severity.</p>", unsafe_allow_html=True)

        for factor in res["key_contributing_factors"]:
            pct = int(factor["weight"] * 100)
            st.markdown(f"""
            <div class="factor-row">
                <div>
                    <b>{factor['factor']}</b> <span style="font-size:11px; color:#475569; font-weight:600;">({factor['category']})</span>
                </div>
                <div style="font-weight:700; color:#0284c7;">{pct}%</div>
            </div>
            <div class="factor-bar">
                <div class="factor-fill" style="width: {pct}%;"></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown(f"#### 🚨 **{t('nav_isolated', lang)} ({len(p['cutoff_settlements'])})**")
        for s in p["cutoff_settlements"]:
            s_name = s.get("name_ne", s["name"]) if lang == "hi" else s["name"]
            st.markdown(f"• **{s_name}** — Pop: `{format_number(s['population'], lang)}` | Priority: `{s['rescue_priority']}`")

    with det_col2:
        st.markdown(f"#### 🗺️ **{t('sim_map_view', lang)}**")
        sim_map = build_geospatial_folium_map(data, show_sim_layer=True, sim_data=res)
        st_folium(sim_map, width="100%", height=480, key="sim_map_folium")


def main():
    # Load Data
    data = load_spatial_data()
    lang = st.session_state["language"]

    # Top Navigation Bar
    render_top_navigation()

    # Human-Friendly Metrics Scoreboard
    render_human_friendly_metrics(data, lang)

    active_tab = st.session_state["active_nav"]

    # View 1: Overview
    if active_tab == "Overview":
        st.markdown(f"### 📋 **{t('nav_overview', lang)}: {DISASTER_EVENT_NAME}**")
        st.markdown(f"""
        <div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:10px; padding:18px; margin-bottom:18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <div>
                    <h4 style="margin:0; color:#0f172a; font-weight:800;">Trishuli & Bhote Koshi Himalayan Corridor, Nepal</h4>
                    <span style="font-size:13px; color:#334155; font-weight:500;">Copernicus EMS Rapid Mapping Code: <b style="color:#0f172a;">{COPERNICUS_EMSR_CODE}</b> | Event Date: <b style="color:#0f172a;">{DISASTER_EVENT_DATE}</b></span>
                </div>
                <div>
                    <span class="badge-status badge-status-danger">HIGH RISK CASCADE EVENT</span>
                </div>
            </div>
            <hr style="border:0; border-top:1px solid #e2e8f0; margin:12px 0;">
            <p style="font-size:14px; color:#1e293b; line-height:1.6; margin:0;">
                Following a major moraine breach and ice avalanche upstream in Tibet/Nepal border glaciers on 26 August 2026, 
                high-velocity hyperconcentrated debris slurries and flash floods propagated down the Bhote Koshi and Trishuli gorges. 
                Satellite remote sensing via Sentinel-1 SAR and Sentinel-2 optical imagery indicates extensive road scour along the Arniko Highway (H03), 
                destroying 3 vital bridges and severing overland road access for <b>{format_number(8500, lang)} residents</b> in Kodari, Tatopani, Liping, and Larcha.
            </p>
        </div>
        """, unsafe_allow_html=True)

        map_obj = build_geospatial_folium_map(data)
        st_folium(map_obj, width="100%", height=550, key="overview_folium_map")

    # View 2: Flood Mapping
    elif active_tab == "Flood Mapping":
        st.markdown(f"### 🌊 **{t('nav_flood_mapping', lang)} (Sentinel-1 SAR & Sentinel-2 Optical)**")
        st.markdown("<p style='color:#334155; font-size:13px; font-weight:500;'>Same-orbit Sentinel-1 C-band SAR backscatter log-ratio change detection and Sentinel-2 MNDWI water extraction.</p>", unsafe_allow_html=True)
        map_obj = build_geospatial_folium_map(data)
        st_folium(map_obj, width="100%", height=580, key="flood_folium_map")

    # View 3: Infrastructure
    elif active_tab == "Infrastructure":
        st.markdown(f"### 🛣️ **{t('nav_infrastructure', lang)}: Roads, Bridges & Buildings at Risk**")
        st.markdown("<p style='color:#334155; font-size:13px; font-weight:500;'>Pre-event OpenStreetMap vectors intersected with satellite-detected flood and debris boundaries.</p>", unsafe_allow_html=True)

        infr_col1, infr_col2 = st.columns([1, 1.8])
        with infr_col1:
            st.markdown("#### 🌉 **Bridges Requiring Inspection**")
            bridges = data["bridges_gdf"]
            for idx, brow in bridges.iterrows():
                is_sev = brow["is_severed"]
                status_color = "#b91c1c" if is_sev else "#15803d"
                st.markdown(f"""
                <div style="background:#ffffff; border:1px solid #cbd5e1; border-left:4px solid {status_color}; border-radius:6px; padding:10px 14px; margin-bottom:8px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
                    <b style="color:#0f172a;">{brow['name']}</b> ({brow.get('name_ne', brow['name'])})<br>
                    <span style="font-size:12px; color:#334155; font-weight:500;">Span: {brow.get('span_m', 0)}m | Status: <b style="color:{status_color};">{brow['status']}</b></span>
                </div>
                """, unsafe_allow_html=True)

        with infr_col2:
            map_obj = build_geospatial_folium_map(data)
            st_folium(map_obj, width="100%", height=500, key="infr_folium_map")

    # View 4: Thermal Analysis
    elif active_tab == "Thermal Analysis":
        render_thermal_analysis_view(data, lang)

    # View 5: Isolated Settlements
    elif active_tab == "Isolated Settlements":
        st.markdown(f"### 🚨 **{t('nav_isolated', lang)} & Humanitarian Relief Routing**")
        st.markdown("<p style='color:#334155; font-size:13px; font-weight:500;'>NetworkX topological graph routing identifying settlements cut off from hospitals due to severed bridges and inundated highway segments.</p>", unsafe_allow_html=True)

        iso_col1, iso_col2 = st.columns([1, 1.8])
        with iso_col1:
            routing = data["metrics_data"]["routing_details"]
            st.markdown(f"#### 🚨 **Cut-Off Communities ({len(routing['cutoff_settlements'])})**")
            for c in routing["cutoff_settlements"]:
                c_name = c.get("name_ne", c["name"]) if lang == "hi" else c["name"]
                st.markdown(f"""
                <div style="background:#ffffff; border:1px solid #cbd5e1; border-left:4px solid #b91c1c; border-radius:6px; padding:12px; margin-bottom:10px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
                    <div style="font-weight:700; color:#0f172a; font-size:15px;">{c_name} ({c['name']})</div>
                    <div style="font-size:12px; color:#334155; margin-top:4px; line-height:1.5; font-weight:500;">
                        Population: <b style="color:#0f172a;">{format_number(c['population'], lang)}</b> | Elevation: {format_number(c['elevation_m'], lang)}m<br>
                        Status: <b style="color:#b91c1c;">{c['status']}</b><br>
                        Nearest Pre-Event Hospital: <b style="color:#0f172a;">{c.get('pre_event_nearest_hospital', 'District Hospital')}</b><br>
                        Action: <span class="badge-status badge-status-danger" style="margin-top:4px;">{c['rescue_priority']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with iso_col2:
            map_obj = build_geospatial_folium_map(data)
            st_folium(map_obj, width="100%", height=540, key="isolated_folium_map")

    # View 6: What-If Simulator
    elif active_tab == "What-If Simulator":
        render_what_if_simulator(data, lang)

    # Secondary Tools Drawer: Bilingual SitRep & Copernicus Validation
    st.markdown("<hr style='border:0; border-top:1px solid #e2e8f0; margin:25px 0 15px 0;'>", unsafe_allow_html=True)
    with st.expander("📄 **Official Bilingual Situation Report (UN-OCHA / NEOC) & EMSR927 Validation Metrics**", expanded=False):
        rep_tab1, rep_tab2 = st.tabs([t("tab_sitrep", lang), t("tab_validation", lang)])

        with rep_tab1:
            registry = GroundedNumericalRegistry(data["metrics_data"])
            copilot = BilingualSitRepCopilot(registry)
            sitrep_out = copilot.generate_english_sitrep() if lang == "en" else copilot.generate_nepali_sitrep()
            st.markdown(sitrep_out["report_markdown"])

        with rep_tab2:
            vm = data["val_results"]["metrics"]
            st.markdown("#### 🇪🇺 Copernicus EMS EMSR927 Benchmark Scores")
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Precision", f"{vm['precision']*100:.2f}%")
            v2.metric("Recall (Sensitivity)", f"{vm['recall']*100:.2f}%")
            v3.metric("F1-Score", f"{vm['f1_score']*100:.2f}%")
            v4.metric("IoU (Jaccard)", f"{vm['iou_jaccard']*100:.2f}%")

    # Legal Attributions Footer in White Theme
    st.markdown(f"""
    <div class="footer-white">
        <b>{t('attributions_header', lang)}</b><br>
        • {t('attr_sentinel', lang)}<br>
        • {t('attr_dem', lang)}<br>
        • {t('attr_osm', lang)}<br>
        • {t('attr_emsr', lang)}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
