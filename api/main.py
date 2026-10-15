"""
FloodWatch-Himalaya: FastAPI Predictive Model & Spatial Backend Service.
Provides RESTful endpoints for:
- Model health and validation status
- Scenario validation
- Running What-If flood impact simulations
- Querying scenario presets
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from ml.config import SCENARIO_PRESETS, METADATA_PATH
from ml.predict import WhatIfFloodPredictor

app = FastAPI(
    title="FloodWatch-Himalaya Geospatial & ML API",
    description="Multimodal spaceborne flood impact forecasting and humanitarian routing engine.",
    version="2.6.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and thermal job instances
predictor: Optional[WhatIfFloodPredictor] = None
thermal_jobs: Dict[str, Dict[str, Any]] = {}

from pipeline.thermal_pipeline import (
    ThermalDataService,
    ThermalRasterProcessor,
    THERMAL_AOI_PRESETS,
    AOI_BBOX_WGS84,
    LEGAL_ATTRIBUTIONS
)
import uuid


@app.on_event("startup")
def startup_event():
    global predictor
    predictor = WhatIfFloodPredictor()


class ScenarioRequest(BaseModel):
    rainfall_intensity_mmh: float = Field(default=25.0, ge=0.0, le=250.0, description="Rainfall intensity in mm/h")
    rainfall_duration_hrs: float = Field(default=12.0, ge=0.5, le=96.0, description="Rainfall duration in hours")
    surge_height_m: float = Field(default=2.5, ge=0.0, le=15.0, description="Estimated river water-level surge in meters")
    upstream_breach_location: str = Field(default="Upper Bhote Koshi Glacier / Liping", description="Upstream failure location")
    thermal_intensity_k: float = Field(default=3.2, ge=0.0, le=12.0, description="Thermal kinetic heating anomaly in Kelvin")
    propagation_factor: float = Field(default=1.0, ge=0.5, le=3.5, description="Debris runout velocity / propagation factor")
    road_closure_override: bool = Field(default=False, description="Manual road closure override")


@app.get("/health")
def health_check():
    return {"status": "HEALTHY", "service": "FloodWatch-Himalaya API", "version": "2.6.0"}


@app.get("/api/model/status")
def get_model_status():
    """Returns model version, feature names, training metrics, and operational readiness."""
    if not METADATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Models have not been trained yet.")
    import json
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata


@app.get("/api/model/presets")
def get_scenario_presets():
    """Returns validated scenario presets: Baseline, Severe, and Extreme."""
    return SCENARIO_PRESETS


@app.post("/api/model/validate")
def validate_scenario(req: ScenarioRequest):
    """Validates scenario input parameter bounds and physical plausibility."""
    warnings = []
    if req.surge_height_m > 8.0:
        warnings.append("Surge height exceeds historical maximum recorded in Bhote Koshi (8.0m in 2016).")
    if req.rainfall_intensity_mmh > 100.0:
        warnings.append("Rainfall intensity represents an extreme 100-year cloudburst event.")

    return {
        "is_valid": True,
        "warnings": warnings,
        "input_summary": {
            "cumulative_precip_mm": req.rainfall_intensity_mmh * req.rainfall_duration_hrs,
            "surge_height_m": req.surge_height_m,
            "breach_corridor": req.upstream_breach_location
        }
    }


@app.post("/api/model/simulate")
def run_simulation(req: ScenarioRequest):
    """Executes the machine learning prediction pipeline for the scenario."""
    global predictor
    if predictor is None:
        predictor = WhatIfFloodPredictor()

    try:
        results = predictor.predict_scenario(req.dict())
        # Strip GeoDataFrames before returning JSON
        clean_results = {
            "scenario_metadata": results["scenario_metadata"],
            "predictions": results["predictions"],
            "comparison_with_baseline": results["comparison_with_baseline"],
            "key_contributing_factors": results["key_contributing_factors"]
        }
        return clean_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


# ==============================================================================
# Thermal Satellite Analysis REST Endpoints
# ==============================================================================

class ThermalProcessRequest(BaseModel):
    mode: str = Field(default="anomaly", description="'surface_temp' or 'anomaly'")
    satellite: str = Field(default="landsat-c2-l2", description="Satellite collection: landsat-c2-l2 or sentinel-3-slstr-lst-l2-netcdf")
    aoi_bbox: Optional[List[float]] = Field(default=None, description="[min_lon, min_lat, max_lon, max_lat]")
    item_id: Optional[str] = Field(default=None, description="Item ID for single-date surface temperature mode")
    pre_item_id: Optional[str] = Field(default=None, description="Pre-event baseline item ID")
    post_item_id: Optional[str] = Field(default=None, description="Post-event item ID")
    start_date: Optional[str] = Field(default="2024-05-01", description="Search start date YYYY-MM-DD")
    end_date: Optional[str] = Field(default="2026-09-01", description="Search end date YYYY-MM-DD")
    apply_cloud_mask: bool = Field(default=True, description="Whether to mask clouds via QA_PIXEL")


@app.get("/api/thermal/presets")
def get_thermal_presets():
    """Returns geographical AOI presets for the Himalayan disaster corridor."""
    return {
        "presets": {name: list(bbox) for name, bbox in THERMAL_AOI_PRESETS.items()},
        "default": "Bhote Koshi Gorge Corridor (Liping to Barhabise)"
    }


@app.get("/api/thermal/available")
def search_thermal_acquisitions(
    min_lon: float = 85.80,
    min_lat: float = 27.80,
    max_lon: float = 86.05,
    max_lat: float = 28.15,
    start_date: str = "2024-05-01",
    end_date: str = "2026-09-01",
    satellite: str = "landsat-c2-l2",
    max_cloud: float = 80.0,
    limit: int = 10
):
    """
    Discovers available thermal satellite acquisitions across the selected area of interest
    using Microsoft Planetary Computer STAC (with USGS/NASA Landsat 8/9 C2 L2 & Sentinel-3).
    """
    bbox = (min_lon, min_lat, max_lon, max_lat)
    try:
        items = ThermalDataService.search_acquisitions(
            aoi_bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            satellite=satellite,
            max_cloud_cover=max_cloud,
            limit=limit
        )
        # Clean items for JSON response (omit unneeded large raw GeoJSON objects)
        clean_items = []
        for it in items:
            clean_items.append({
                "id": it["id"],
                "date": it["date"],
                "datetime": it.get("datetime", it["date"]),
                "platform": it["platform"],
                "cloud_cover": it["cloud_cover"],
                "is_live": it.get("is_live", False),
                "is_reference": it.get("is_reference", False),
                "source": it.get("source", "USGS Landsat Collection 2 Level-2 ST")
            })
        return {
            "total_found": len(clean_items),
            "satellite": satellite,
            "aoi_bbox": list(bbox),
            "acquisitions": clean_items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thermal search failed: {str(e)}")


@app.post("/api/thermal/process")
def process_thermal_imagery(req: ThermalProcessRequest):
    """
    Executes physical temperature calibration, cloud quality masking,
    spatial grid reprojection, and temperature anomaly calculations.
    """
    global thermal_jobs
    bbox = tuple(req.aoi_bbox) if req.aoi_bbox and len(req.aoi_bbox) == 4 else AOI_BBOX_WGS84

    job_id = f"thermal_{uuid.uuid4().hex[:8]}"

    try:
        if req.mode == "surface_temp":
            # 1. Surface Temperature Mode
            # Resolve item
            item_arg: Union[Dict[str, Any], str] = {}
            if req.item_id:
                # Search or check reference
                avail = ThermalDataService.search_acquisitions(bbox, start_date=req.start_date, end_date=req.end_date, satellite=req.satellite)
                matched = [it for it in avail if it["id"] == req.item_id]
                item_arg = matched[0] if matched else req.item_id
            else:
                ref_items = ThermalDataService.get_reference_fixtures(bbox, req.satellite)
                item_arg = ref_items[0]

            res = ThermalRasterProcessor.load_surface_temperature(item_arg, aoi_bbox=bbox, apply_cloud_mask=req.apply_cloud_mask)
            overlay_uri, legend = ThermalRasterProcessor.render_overlay_png(res["temp_celsius"], mode="surface_temp")

            thermal_jobs[job_id] = {
                "job_id": job_id,
                "status": "COMPLETED",
                "mode": "surface_temp",
                "satellite": req.satellite,
                "item_id": res["id"],
                "date": res["date"],
                "platform": res["platform"],
                "stats": res["stats"],
                "bounds": res["wgs_bounds"],
                "bounds_geojson": res["bounds_geojson"],
                "overlay_uri": overlay_uri,
                "legend": legend,
                "data_grid": res["temp_celsius"],
                "transform": res["transform"],
                "crs": res["crs"],
                "is_reference": res.get("is_reference", False),
                "attribution": LEGAL_ATTRIBUTIONS.get("copernicus_sentinel", "Landsat data courtesy of the U.S. Geological Survey")
            }

            return {
                "job_id": job_id,
                "status": "COMPLETED",
                "mode": "surface_temp",
                "date": res["date"],
                "stats": res["stats"],
                "legend": legend,
                "bounds": res["wgs_bounds"]
            }

        else:
            # 2. Pre vs Post-Event Anomaly Mode
            ref_items = ThermalDataService.get_reference_fixtures(bbox, req.satellite)
            pre_item: Union[Dict[str, Any], str] = ref_items[0]
            post_item: Union[Dict[str, Any], str] = ref_items[1]

            if req.pre_item_id or req.post_item_id:
                avail = ThermalDataService.search_acquisitions(bbox, start_date=req.start_date, end_date=req.end_date, satellite=req.satellite)
                if req.pre_item_id:
                    m_pre = [it for it in avail if it["id"] == req.pre_item_id]
                    if m_pre:
                        pre_item = m_pre[0]
                if req.post_item_id:
                    m_post = [it for it in avail if it["id"] == req.post_item_id]
                    if m_post:
                        post_item = m_post[0]

            pre_res = ThermalRasterProcessor.load_surface_temperature(pre_item, aoi_bbox=bbox, apply_cloud_mask=req.apply_cloud_mask)
            post_res = ThermalRasterProcessor.load_surface_temperature(post_item, aoi_bbox=bbox, apply_cloud_mask=req.apply_cloud_mask)

            anomaly_res = ThermalRasterProcessor.compute_anomaly(pre_res, post_res)
            overlay_uri, legend = ThermalRasterProcessor.render_overlay_png(anomaly_res["delta_temp"], mode="anomaly")

            thermal_jobs[job_id] = {
                "job_id": job_id,
                "status": "COMPLETED",
                "mode": "anomaly",
                "satellite": req.satellite,
                "pre_id": anomaly_res["pre_id"],
                "pre_date": anomaly_res["pre_date"],
                "post_id": anomaly_res["post_id"],
                "post_date": anomaly_res["post_date"],
                "stats": anomaly_res["stats"],
                "bounds": anomaly_res["wgs_bounds"],
                "bounds_geojson": anomaly_res["bounds_geojson"],
                "overlay_uri": overlay_uri,
                "legend": legend,
                "data_grid": anomaly_res["delta_temp"],
                "transform": anomaly_res["transform"],
                "crs": anomaly_res["crs"],
                "attribution": "USGS/NASA Landsat 8/9 Collection 2 Level-2 Surface Temperature ST_B10"
            }

            return {
                "job_id": job_id,
                "status": "COMPLETED",
                "mode": "anomaly",
                "pre_date": anomaly_res["pre_date"],
                "post_date": anomaly_res["post_date"],
                "stats": anomaly_res["stats"],
                "legend": legend,
                "bounds": anomaly_res["wgs_bounds"]
            }

    except Exception as e:
        thermal_jobs[job_id] = {
            "job_id": job_id,
            "status": "FAILED",
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=f"Thermal processing failed: {str(e)}")


@app.get("/api/thermal/status/{job_id}")
def get_thermal_job_status(job_id: str):
    """Checks the status of an asynchronous or executed thermal processing job."""
    global thermal_jobs
    if job_id not in thermal_jobs:
        raise HTTPException(status_code=404, detail=f"Thermal job '{job_id}' not found.")
    job = thermal_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "mode": job.get("mode"),
        "error": job.get("error")
    }


@app.get("/api/thermal/metadata/{job_id}")
def get_thermal_job_metadata(job_id: str):
    """Retrieves detailed statistical metadata and physical observations for a thermal job."""
    global thermal_jobs
    if job_id not in thermal_jobs:
        raise HTTPException(status_code=404, detail=f"Thermal job '{job_id}' not found.")
    job = thermal_jobs[job_id]
    if job["status"] == "FAILED":
        raise HTTPException(status_code=400, detail=job.get("error", "Job failed."))

    return {
        "job_id": job_id,
        "mode": job["mode"],
        "satellite": job.get("satellite"),
        "date": job.get("date"),
        "pre_date": job.get("pre_date"),
        "post_date": job.get("post_date"),
        "stats": job["stats"],
        "bounds": job["bounds"],
        "bounds_geojson": job["bounds_geojson"],
        "legend": job["legend"],
        "attribution": job.get("attribution", "")
    }


@app.get("/api/thermal/layer/{job_id}")
def get_thermal_layer(job_id: str):
    """Retrieves georeferenced overlay image and bounds for Leaflet/Folium map rendering."""
    global thermal_jobs
    if job_id not in thermal_jobs:
        raise HTTPException(status_code=404, detail=f"Thermal job '{job_id}' not found.")
    job = thermal_jobs[job_id]
    if job["status"] == "FAILED":
        raise HTTPException(status_code=400, detail=job.get("error", "Job failed."))

    return {
        "job_id": job_id,
        "overlay_uri": job["overlay_uri"],
        "bounds": job["bounds"],
        "bounds_geojson": job["bounds_geojson"],
        "legend": job["legend"]
    }


@app.get("/api/thermal/point")
def probe_thermal_point(job_id: str, lat: float, lon: float):
    """Probes the calibrated thermal raster at a specific latitude and longitude."""
    global thermal_jobs
    if job_id not in thermal_jobs:
        raise HTTPException(status_code=404, detail=f"Thermal job '{job_id}' not found.")
    job = thermal_jobs[job_id]
    if job["status"] != "COMPLETED":
        raise HTTPException(status_code=400, detail="Job has not completed successfully.")

    val = ThermalRasterProcessor.sample_point_temperature(
        job["data_grid"],
        job["transform"],
        job["crs"],
        lat=lat,
        lon=lon
    )
    return {
        "job_id": job_id,
        "lat": lat,
        "lon": lon,
        "mode": job["mode"],
        "unit": "°C" if job["mode"] == "surface_temp" else "ΔK (°C)",
        "value": val
    }

