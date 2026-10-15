"""
FloodWatch-Himalaya: Master Orchestrator CLI.
Executes the complete end-to-end spaceborne flood mapping and humanitarian routing pipeline
for the August 2026 Trishuli / Bhote Koshi disaster.
"""

import sys
import json
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DISASTER_EVENT_NAME, DISASTER_EVENT_DATE, COPERNICUS_EMSR_CODE, LEGAL_ATTRIBUTIONS
from data.generate_fixtures import create_trishuli_bhote_koshi_fixtures
from pipeline.ingest_and_change import MultimodalFusionEngine
from pipeline.network_routing import run_network_routing_analysis
from pipeline.validation_emsr import CopernicusEMSRValidator
from copilot.rag_sitrep import BilingualSitRepCopilot


def run_full_pipeline():
    print("=" * 80)
    print(f"🛰️  FLOODWATCH-HIMALAYA: END-TO-END SPACEBORNE RESPONSE PIPELINE")
    print(f"    Event: {DISASTER_EVENT_NAME} ({DISASTER_EVENT_DATE})")
    print(f"    Activation: Copernicus EMS {COPERNICUS_EMSR_CODE}")
    print("=" * 80)

    # 1. Initialize Fixtures and Pre-event Data
    print("\n[STEP 1/5] Initializing Pre-event OSM and Spatial Fixtures...")
    fixtures = create_trishuli_bhote_koshi_fixtures()
    print(f"  ✓ Created {len(fixtures)} spatial fixture layers in data/sample_fixtures/")

    # 2. Multimodal Spaceborne Ingestion & Change Detection
    print("\n[STEP 2/5] Executing Sentinel-1 SAR & Thermal LST Multimodal Ingestion...")
    engine = MultimodalFusionEngine()
    ingest_res = engine.execute_pipeline()
    print(f"  ✓ Sentinel-1 SAR Orbit Track {ingest_res['sar_metrics']['orbit_track']} calibrated & Lee filtered")
    print(f"  ✓ Inundated Water Surface: {ingest_res['total_water_area_sqkm']:.2f} km²")
    print(f"  ✓ Friction-Heated Debris Flow: {ingest_res['total_debris_area_sqkm']:.2f} km²")
    print(f"  ✓ Peak Thermal LST Anomaly: +{ingest_res['thermal_metrics']['max_lst_anomaly_kelvin']:.2f} K")

    # 3. Vector Overlay & Graph Cut-Off Routing
    print("\n[STEP 3/5] Performing OSM Vector Overlays & NetworkX Cut-off Routing...")
    routing_res = run_network_routing_analysis()
    m = routing_res["metrics"]
    print(f"  ✓ Cut-off Settlements Flagged: {m['cut_off_settlement_count']} isolated communities")
    print(f"  ✓ Stranded Population Count: {m['total_stranded_population']:,} persons")
    print(f"  ✓ Severed Bridge Crossings: {m['severed_bridges_count']} bridges")
    print(f"  ✓ Submerged Road Network: {m['flooded_road_length_km']} km ({m['flooded_road_length_meters']} m)")
    print(f"  ✓ Damaged Building Footprints: {m['damaged_buildings_count']} structures ({m['damaged_buildings_area_sqm']:,} m²)")

    # 4. Copernicus EMSR927 Validation
    print("\n[STEP 4/5] Computing Copernicus EMS EMSR927 Benchmark Validation...")
    validator = CopernicusEMSRValidator()
    val_res = validator.compute_metrics()
    vm = val_res["metrics"]
    print(f"  ✓ Precision: {vm['precision'] * 100:.2f}% | Recall: {vm['recall'] * 100:.2f}%")
    print(f"  ✓ F1-Score: {vm['f1_score'] * 100:.2f}% | IoU (Jaccard): {vm['iou_jaccard'] * 100:.2f}%")
    print(f"  ✓ Assessment: {val_res['interpretation']['benchmark_rating']}")

    # 5. Bilingual Grounded SitRep Generation
    print("\n[STEP 5/6] Generating Grounded Situation Reports (English & Nepali)...")
    copilot = BilingualSitRepCopilot()
    en_rep = copilot.generate_english_sitrep()
    ne_rep = copilot.generate_nepali_sitrep()
    print(f"  ✓ English SitRep: {en_rep['audit']['audit_verdict']} ({en_rep['audit']['verified_count']} numbers verified)")
    print(f"  ✓ Nepali SitRep:  {ne_rep['audit']['audit_verdict']} ({ne_rep['audit']['verified_count']} numbers verified)")

    # 6. AI/ML What-If Scenario Prediction
    print("\n[STEP 6/6] Executing Trained AI/ML What-If Predictive Simulation...")
    from ml.predict import WhatIfFloodPredictor
    from ml.config import SCENARIO_PRESETS
    predictor = WhatIfFloodPredictor()
    sim_res = predictor.predict_scenario(SCENARIO_PRESETS["severe"])
    sp = sim_res["predictions"]
    sc = sim_res["comparison_with_baseline"]
    print(f"  ✓ Severe Preset Prediction: {sp['inundated_area_sqkm']} km² inundation extent")
    print(f"  ✓ Estimated Impacted Buildings: {sp['impacted_buildings_estimate']} (CI: {sp['impacted_buildings_ci']})")
    print(f"  ✓ Impassable Roads: {sp['flooded_road_length_km']} km | Bridges at Risk: {sp['bridges_requiring_inspection_count']}")
    print(f"  ✓ Predicted Cut-off Communities: {sp['cutoff_settlement_count']} settlements ({sp['stranded_population_count']:,} residents)")

    print("\n" + "=" * 80)
    print("MANDATORY SENSOR & DATA ATTRIBUTIONS:")
    for key, text in LEGAL_ATTRIBUTIONS.items():
        print(f"  • {text}")
    print("=" * 80)
    print("✨ Pipeline execution complete! To launch the interactive dashboard, run:")
    print("    streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    run_full_pipeline()
