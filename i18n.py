"""
FloodWatch-Himalaya: Centralized Internationalization (i18n) Module.
Provides comprehensive, high-fidelity English and Hindi (हिन्दी) translation dictionaries
with proper Devanagari typography, technical disaster terms, tooltips, and localized formatting.
"""

from typing import Dict, Any

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        # Navigation
        "nav_brand": "FLOOD INTELLIGENCE",
        "nav_subtitle": "Spaceborne Disaster Operations | Nepal Track B",
        "nav_overview": "Overview",
        "nav_flood_mapping": "Flood Mapping",
        "nav_infrastructure": "Infrastructure",
        "nav_thermal": "Thermal Analysis",
        "nav_isolated": "Isolated Settlements",
        "nav_simulator": "What-If Simulator",
        "nav_user_role": "Rescue Ops Lead",
        "nav_notifications": "Alerts",
        "nav_lang_toggle": "Language",

        # Header & Badges
        "status_severe": "ACTIVE GLOF / FLASH FLOOD",
        "status_monitoring": "MONITORING ACTIVE",
        "badge_copernicus": "COP-EMS EMSR927",
        "badge_grounded": "ZERO-HALLUCINATION GROUNDING ACTIVE",
        "badge_prediction": "AI/ML PREDICTION (UNVERIFIED)",
        "badge_detected": "SATELLITE DETECTED",
        "badge_estimated": "GEOSPATIAL ESTIMATE",
        "badge_confirmed": "CONFIRMED GROUND TRUTH",

        # Dashboard Metric Cards (Human-Friendly)
        "metric_affected_area": "Flood-Affected Area",
        "metric_affected_area_sub": "Water & debris footprint detected",
        "metric_affected_area_tooltip": "Total land surface area where flood water inundation or debris slurry deposition was detected by Sentinel-1 SAR and Sentinel-2 optical sensors.",

        "metric_impacted_buildings": "Potentially Impacted Buildings",
        "metric_impacted_buildings_sub": "Structures in hazard zone",
        "metric_impacted_buildings_tooltip": "Pre-event OpenStreetMap building footprints intersecting detected flood or debris channels. Requires on-the-ground damage confirmation.",

        "metric_roads_risk": "Roads at Risk",
        "metric_roads_risk_sub": "Arniko & Pasang Lhamu highways",
        "metric_roads_risk_tooltip": "Total length of highway segments submerged by flood waters or blocked by rock/mud debris, rendering overland transit impossible.",

        "metric_bridges_inspection": "Bridges Requiring Inspection",
        "metric_bridges_inspection_sub": "Bridge spans within scour buffer",
        "metric_bridges_inspection_tooltip": "Bridge structures directly intersected by flood torrents or within 150m scour buffer, requiring urgent structural integrity engineering inspections.",

        "metric_isolated_settlements": "Potentially Isolated Settlements",
        "metric_isolated_settlements_sub": "Severed from regional hospitals",
        "metric_isolated_settlements_tooltip": "Communities with zero passable overland road connections to operational healthcare facilities or district staging hubs according to NetworkX topological routing.",

        "metric_thermal_anomalies": "Thermal Anomalies",
        "metric_thermal_anomalies_sub": "Peak kinetic friction heating",
        "metric_thermal_anomalies_tooltip": "Positive Land Surface Temperature (LST) anomaly extracted from Landsat-8/9 TIRS and Sentinel-3 SLSTR, caused by internal shear friction of rock-ice debris slurries.",

        "metric_severity": "Overall Disaster Severity",
        "metric_severity_val": "Level-3 Emergency",
        "metric_severity_sub": "High-altitude cascade event",
        "metric_severity_tooltip": "Multimodal classification combining flood extent, bridge collapse count, and population isolation metrics.",

        # Controls & Buttons
        "btn_run_pipeline": "Refresh Spaceborne Feed",
        "btn_run_pipeline_running": "Processing satellite observations...",
        "btn_download_report": "Download SitRep (.md)",
        "btn_run_simulation": "RUN PREDICTIVE SIMULATION",
        "btn_running_simulation": "Executing scikit-learn models...",
        "btn_reset_simulation": "Reset to Default",

        # Simulator Presets & Inputs
        "sim_title": "AI/ML What-If Flood Scenario Simulator",
        "sim_subtitle": "Trained predictive models forecasting inundation probability, infrastructure disruption, and community isolation.",
        "sim_preset_label": "Select Scenario Preset:",
        "preset_baseline": "Baseline Conditions (Current Observations)",
        "preset_severe": "Severe Flood Scenario (+3.5m surge, 45 mm/h rain)",
        "preset_extreme": "Extreme Flood Scenario (+6.0m surge, 85 mm/h rain, dam breach)",
        "sim_inputs_header": "Scenario Parameters & Forcing Inputs",
        "param_rainfall_intensity": "Rainfall Intensity (mm/h)",
        "param_rainfall_duration": "Rainfall Duration (hours)",
        "param_surge_height": "Estimated River Surge / Water-Level Change (meters)",
        "param_breach_location": "Upstream Failure / Breach Corridor",
        "param_thermal_intensity": "Thermal Anomaly Intensity (ΔK)",
        "param_propagation_factor": "Propagation Factor / Slurry Runout Velocity",
        "param_road_closure_override": "Simulate Immediate Highway Closure",

        # Simulator Outputs
        "sim_results_header": "Simulation Results & Predicted Impacts",
        "sim_notice_prediction": "⚠️ NOTE: Predictions generated by trained Random Forest & Gradient Boosting models. These are predictive estimates for planning, not confirmed observations.",
        "sim_stat_predicted_area": "Predicted Inundated Area",
        "sim_stat_predicted_buildings": "Estimated Impacted Buildings",
        "sim_stat_predicted_roads": "Predicted Impassable Roads",
        "sim_stat_predicted_bridges": "Bridges at Risk",
        "sim_stat_predicted_isolated": "Settlements Losing Road Access",
        "sim_stat_model_confidence": "Model Confidence / Uncertainty",
        "sim_key_factors_title": "Key Contributing Risk Factors (Feature Importances)",
        "sim_comparison_title": "Comparison with Baseline Conditions",
        "sim_map_view": "Predicted Inundation Probability Map",

        # Tabs & Sections
        "tab_map": "Tactical Geospatial Map",
        "tab_sitrep": "Bilingual Situation Report",
        "tab_copilot": "Grounded AI Copilot",
        "tab_validation": "EMSR927 Validation",

        # Copilot Q&A
        "copilot_title": "Grounded Disaster Intelligence Copilot",
        "copilot_subtitle": "Responses strictly verified against the spatial database with zero numerical hallucination.",
        "copilot_input_placeholder": "Ask in English or हिन्दी (e.g., How many people are stranded in Tatopani?)",
        "copilot_suggested": "Suggested Questions:",
        "q_stranded": "How many people are stranded in cut-off settlements?",
        "q_bridges": "Which bridges were severed by the debris torrent?",
        "q_roads": "What is the total length of blocked highway segments?",

        # Thermal Analysis
        "thermal_header": "Spaceborne Thermal Mapping & Kinetic Debris Anomaly Analysis",
        "thermal_subtitle": "USGS Landsat 8/9 Collection 2 Level-2 Surface Temperature (ST_B10) & Sentinel-3 SLSTR LST.",
        "thermal_mode_label": "Analysis Mode",
        "thermal_mode_surface": "Land Surface Temperature (LST)",
        "thermal_mode_anomaly": "Pre- vs Post-Event Temperature Anomaly (ΔT)",
        "thermal_aoi_label": "Geographical Area of Interest (AOI)",
        "thermal_satellite_label": "Satellite Sensor Collection",
        "thermal_start_date": "Observation Window Start",
        "thermal_end_date": "Observation Window End",
        "thermal_btn_search": "Search Available Satellite Passes",
        "thermal_btn_process": "EXECUTE THERMAL PROCESSING",
        "thermal_status_ready": "Satellite scene selected. Ready to process physical temperatures.",
        "thermal_status_processing": "Processing spaceborne thermal radiance & calibrating LST...",
        "thermal_status_success": "Thermal analysis completed successfully.",
        "thermal_status_no_data": "No cloud-free thermal scenes found in the selected time window. Switch to regional reference scenes or widen dates.",
        "thermal_legend_lst": "Surface Temperature (°C)",
        "thermal_legend_anomaly": "Temperature Anomaly ΔT (K / °C)",
        "thermal_opacity_label": "Thermal Layer Opacity",
        "thermal_probe_title": "Interactive Point Temperature Inspection",
        "thermal_probe_hint": "Inspect calibrated thermal observations across the river gorge.",
        "thermal_stat_max_anomaly": "Peak Friction Heating (ΔT)",
        "thermal_stat_mean_temp": "Mean Corridor Temperature",
        "thermal_stat_friction_area": "Friction-Heated Debris Zone",
        "thermal_stat_cloud_pct": "Cloud-Masked Ratio",
        "thermal_stat_valid_pixels": "Calibrated Valid Pixels",
        "thermal_scientific_note": "Scientific Caution: Positive thermal anomalies (ΔT ≥ +2.8 K) indicate kinetic shear friction heating along rapid rock-ice avalanche and mudflow shear margins. Negative anomalies reflect cold glacial runoff or cloud shadow. Always corroborate with Sentinel-1 SAR and Sentinel-2 MNDWI observations.",
        "thermal_toggle_overlay": "Overlay Sentinel-1 SAR & Inundation Layers",

        # Legal Attributions
        "attributions_header": "Mandatory Data & Sensor Attributions (Legal Compliance):",
        "attr_sentinel": "Contains modified Copernicus Sentinel data [2026], processed by FloodWatch-Himalaya.",
        "attr_dem": "Produced using Copernicus WorldDEM-30 © DLR e.V. 2026, European Union.",
        "attr_osm": "OSM data © OpenStreetMap contributors, accessed via HeiGIT ohsome API (snapshot < 2026-08-26).",
        "attr_emsr": "Reference validation derived from Copernicus Emergency Management Service [EMSR927]."
    },
    "hi": {
        # Navigation
        "nav_brand": "बाढ़ प्रज्ञान प्रणाली",
        "nav_subtitle": "भू-उपग्रह आपदा परिचालन केंद्र | नेपाल ट्रैक B",
        "nav_overview": "अवलोकन",
        "nav_flood_mapping": "बाढ़ मानचित्रण",
        "nav_infrastructure": "बुनियादी ढाँचा",
        "nav_thermal": "तापीय विश्लेषण",
        "nav_isolated": "संपर्कविहीन बस्तियाँ",
        "nav_simulator": "आकस्मिक बाढ़ सिम्युलेटर",
        "nav_user_role": "आपदा बचाव प्रमुख",
        "nav_notifications": "चेतावनी",
        "nav_lang_toggle": "भाषा",

        # Header & Badges
        "status_severe": "सक्रिय हिमनदी-विस्फोट / भीषण बाढ़",
        "status_monitoring": "निगरानी सक्रिय",
        "badge_copernicus": "सीओपी-ईएमएस EMSR927",
        "badge_grounded": "सत्यापित सांख्यिकीय सटीकता सक्रिय",
        "badge_prediction": "एआई/एमएल पूर्वानुमान (असत्यापित)",
        "badge_detected": "उपग्रह द्वारा संवेदित",
        "badge_estimated": "भू-स्थानिक अनुमानित",
        "badge_confirmed": "प्रमाणित ज़मीनी तथ्य",

        # Dashboard Metric Cards (Human-Friendly)
        "metric_affected_area": "बाढ़ प्रभावित क्षेत्र",
        "metric_affected_area_sub": "जलमग्न व मलबा फैलाव क्षेत्र",
        "metric_affected_area_tooltip": "सेंटिनल-१ एसएआर और सेंटिनल-२ ऑप्टिकल संवेदकों द्वारा पता लगाया गया कुल जलमग्न और मलबे से प्रभावित भूभाग।",

        "metric_impacted_buildings": "संभावित प्रभावित भवन",
        "metric_impacted_buildings_sub": "जोखिम क्षेत्र में स्थित संरचनाएँ",
        "metric_impacted_buildings_tooltip": "ओपनस्ट्रीटमैप के भवन जो संभावित बाढ़ या मलबे के बहाव क्षेत्र में आते हैं। ज़मीनी सत्यापन प्रतीक्षित है।",

        "metric_roads_risk": "जोखिम में सड़कें",
        "metric_roads_risk_sub": "अरनिको व पासाङ ल्हामु राजमार्ग",
        "metric_roads_risk_tooltip": "बाढ़ के पानी अथवा भूस्खलन मलबे के कारण अवरुद्ध मुख्य राजमार्गों की कुल लंबाई, जहाँ आवागमन बाधित है।",

        "metric_bridges_inspection": "निरीक्षण योग्य पुल",
        "metric_bridges_inspection_sub": "धारा कटाव क्षेत्र में स्थित पुल",
        "metric_bridges_inspection_tooltip": "भीषण जलप्रवाह के सीधे मार्ग में या १५० मीटर कटाव परिधि में आने वाले पुल, जिनका तात्कालिक संरचनात्मक निरीक्षण आवश्यक है।",

        "metric_isolated_settlements": "संभावित संपर्कविहीन बस्तियाँ",
        "metric_isolated_settlements_sub": "अस्पतालों से सड़क संपर्क कटा",
        "metric_isolated_settlements_tooltip": "वे समुदाय जिनका नेटवर्क विश्लेषण के अनुसार निकटतम ज़िला अस्पताल या राहत केंद्र से सड़क संपर्क पूर्णतः टूट चुका है।",

        "metric_thermal_anomalies": "तापीय विसंगतियाँ",
        "metric_thermal_anomalies_sub": "घर्षण जनित तापीय वृद्धि",
        "metric_thermal_anomalies_tooltip": "लैंडसैट-८/९ टीआईआरएस तथा सेंटिनल-३ एसएलएसटीआर द्वारा दर्ज भू-सतह तापमान विसंगति, जो मलबे के घर्षण से उत्पन्न हुई है।",

        "metric_severity": "समग्र आपदा गंभीरता",
        "metric_severity_val": "स्तर-३ आपातकाल",
        "metric_severity_sub": "उच्च-पर्वतीय बहु-आपदा घटना",
        "metric_severity_tooltip": "बाढ़ विस्तार, पुलों की क्षति और आबादी विच्छेदन के आधार पर निर्धारित समग्र आपातकालीन स्तर।",

        # Controls & Buttons
        "btn_run_pipeline": "उपग्रह आँकड़े ताज़ा करें",
        "btn_run_pipeline_running": "उपग्रह अवलोकनों का संसाधन हो रहा है...",
        "btn_download_report": "प्रतिवेदन डाउनलोड करें (.md)",
        "btn_run_simulation": "पूर्वानुमानित सिमुलेशन चलाएँ",
        "btn_running_simulation": "मशीन लर्निंग मॉडल निष्पादित हो रहे हैं...",
        "btn_reset_simulation": "मूल स्थिति पर लाएँ",

        # Simulator Presets & Inputs
        "sim_title": "एआई/एमएल बाढ़ परिदृश्य सिम्युलेटर (What-If)",
        "sim_subtitle": "प्रशिक्षित प्रेडिक्टिव मॉडल द्वारा जलप्रवाह संभावना, बुनियादी ढाँचा क्षति और बस्ती विच्छेदन का पूर्वानुमान।",
        "sim_preset_label": "परिदृश्य पूर्व-निर्धारण (Preset) चुनें:",
        "preset_baseline": "सामान्य स्थिति (वर्तमान अवलोकन)",
        "preset_severe": "गंभीर बाढ़ परिदृश्य (+३.५ मी. जलवृद्धि, ४५ मिमी/घं वर्षा)",
        "preset_extreme": "अत्यंत गंभीर बाढ़ परिदृश्य (+६.० मी. जलवृद्धि, ८५ मिमी/घं वर्षा, तटबंध टूटना)",
        "sim_inputs_header": "परिदृश्य चर एवं इनपुट पैरामीटर",
        "param_rainfall_intensity": "वर्षा की तीव्रता (मिमी/घंटा)",
        "param_rainfall_duration": "वर्षा की अवधि (घंटे)",
        "param_surge_height": "अनुमानित नदी जलस्तर वृद्धि (मीटर)",
        "param_breach_location": "ऊपरी बहाव विखंडन / झील क्षेत्र",
        "param_thermal_intensity": "तापीय विसंगति तीव्रता (ΔK)",
        "param_propagation_factor": "मलबा प्रसार गति / संचरण गुणांक",
        "param_road_closure_override": "सड़क बंद करने का अनुकरण करें",

        # Simulator Outputs
        "sim_results_header": "सिमुलेशन परिणाम एवं पूर्वानुमानित प्रभाव",
        "sim_notice_prediction": "⚠️ सूचना: ये परिणाम प्रशिक्षित रैंडम फॉरेस्ट और ग्रेडिएंट बूस्टिंग मॉडल द्वारा निकाले गए पूर्वानुमान हैं। इन्हें राहत योजना हेतु उपयोग करें, न कि पुष्ट सत्य के रूप में।",
        "sim_stat_predicted_area": "पूर्वानुमानित बाढ़ क्षेत्र",
        "sim_stat_predicted_buildings": "अनुमानित प्रभावित भवन",
        "sim_stat_predicted_roads": "जोखिमग्रस्त अवरुद्ध सड़कें",
        "sim_stat_predicted_bridges": "जोखिम में पुल",
        "sim_stat_predicted_isolated": "संपर्क खोने वाली बस्तियाँ",
        "sim_stat_model_confidence": "मॉडल विश्वसनीयता / अनिश्चितता",
        "sim_key_factors_title": "प्रमुख योगदानकर्ता जोखिम कारक (फीचर महत्व)",
        "sim_comparison_title": "सामान्य स्थिति से तुलना",
        "sim_map_view": "पूर्वानुमानित जलमग्न संभावना मानचित्र",

        # Tabs & Sections
        "tab_map": "रणनीतिक भू-स्थानिक मानचित्र",
        "tab_sitrep": "द्विभाषी वस्तुस्थिति प्रतिवेदन",
        "tab_copilot": "प्रमाणित एआई कोपायलट",
        "tab_validation": "EMSR927 सत्यापन",

        # Copilot Q&A
        "copilot_title": "प्रमाणित आपदा प्रज्ञान कोपायलट",
        "copilot_subtitle": "सभी उत्तर भू-स्थानिक डेटाबेस से गणितीय रूप से सत्यापित हैं (शून्य भ्रम / Zero Hallucination)।",
        "copilot_input_placeholder": "अंग्रेजी या हिन्दी में प्रश्न पूछें (उदा. तातोपानी में कितने लोग फंसे हैं?)",
        "copilot_suggested": "सुझाए गए प्रश्न:",
        "q_stranded": "संपर्कविहीन बस्तियों में कुल कितने नागरिक फंसे हैं?",
        "q_bridges": "मलबा बहाव से कौन-कौन से पुल क्षतिग्रस्त हुए हैं?",
        "q_roads": "कितने किलोमीटर राजमार्ग अवरुद्ध हुआ है?",

        # Thermal Analysis
        "thermal_header": "अन्तरिक्षीय तापीय विश्लेषण एवं घर्षण विसङ्गति संवेदन",
        "thermal_subtitle": "यूएसजीएस लैंडसैट ८/९ स्तर-२ सतह तापमान (ST_B10) एवं सेंटिनल-३ SLSTR LST।",
        "thermal_mode_label": "विश्लेषण विधा",
        "thermal_mode_surface": "धरातलीय सतह तापमान (LST)",
        "thermal_mode_anomaly": "घटनापूर्व बनाम घटनापश्चात् तापमान विसंगति (ΔT)",
        "thermal_aoi_label": "अध्ययन क्षेत्र (AOI)",
        "thermal_satellite_label": "उपग्रह सेंसर संग्रह",
        "thermal_start_date": "निगरानी प्रारंभ तिथि",
        "thermal_end_date": "निगरानी समाप्ति तिथि",
        "thermal_btn_search": "उपलब्ध उपग्रह पास खोजें",
        "thermal_btn_process": "थर्मल डेटा प्रसंस्करण निष्पादित करें",
        "thermal_status_ready": "उपग्रह दृश्य चयनित। तापमान अंशांकन हेतु तैयार।",
        "thermal_status_processing": "अन्तरिक्षीय तापीय डेटा का अंशांकन एवं प्रसंस्करण जारी...",
        "thermal_status_success": "तापीय विश्लेषण सफलतापूर्वक पूर्ण हुआ।",
        "thermal_status_no_data": "चयनित समय-सीमा में बादल-रहित दृश्य उपलब्ध नहीं है। कृपया क्षेत्रीय संदर्भ डेटा चुनें।",
        "thermal_legend_lst": "सतह तापमान (°C)",
        "thermal_legend_anomaly": "तापमान विसंगति ΔT (K / °C)",
        "thermal_opacity_label": "थर्मल परत पारदर्शिता (Opacity)",
        "thermal_probe_title": "इंटरैक्टिव बिंदु तापमान परीक्षण",
        "thermal_probe_hint": "नदी घाटी में किसी भी बिंदु का तापमान जांचें।",
        "thermal_stat_max_anomaly": "चरम घर्षण तापन (ΔT)",
        "thermal_stat_mean_temp": "औसत घाटी तापमान",
        "thermal_stat_friction_area": "घर्षण-तप्त मलबा क्षेत्र",
        "thermal_stat_cloud_pct": "बादल-आच्छादित अनुपात",
        "thermal_stat_valid_pixels": "अंशांकित मान्य पिक्सेल",
        "thermal_scientific_note": "वैज्ञानिक चेतावनी: सकारात्मक तापीय विसंगति (ΔT ≥ +२.८ K) तीव्र चट्टान-बर्फ हिमस्खलन और मलबा बहाव में घर्षण तापन दर्शाती है। इसे हमेशा सेंटिनल-१ एसएआर और सेंटिनल-२ एमएनडीडब्ल्यूआई से सत्यापित किया जाना चाहिए।",
        "thermal_toggle_overlay": "सेंटिनल-१ एसएआर एवं बाढ़ मानचित्रण परतें जोड़ें",

        # Legal Attributions
        "attributions_header": "अनिवार्य डेटा एवं सेंसर आभार (विधिक अनुपालन):",
        "attr_sentinel": "संशोधित कॉपरनिकस सेंटिनल डेटा [२०२६] शामिल है, जिसे फ्लडवॉच-हिमालय द्वारा संसाधित किया गया है।",
        "attr_dem": "कॉपरनिकस वर्ल्डडीईएम-३० © डीएलआर ई.वी. २०२६, यूरोपीय संघ का उपयोग करके निर्मित।",
        "attr_osm": "ओपनस्ट्रीटमैप डेटा © ओपनस्ट्रीटमैप योगदानकर्ता, हाईजीआईटी ओहसम एपीआई द्वारा अधिग्रहित (< २६ अग. २०२६)।",
        "attr_emsr": "कॉपरनिकस आपातकालीन प्रबंधन सेवा [EMSR927] से प्राप्त संदर्भ सत्यापन।"
    }
}


def t(key: str, lang: str = "en") -> str:
    """Retrieves translation for the given key in the specified language (default 'en')."""
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return lang_dict.get(key, TRANSLATIONS["en"].get(key, key))


def format_number(val: Any, lang: str = "en") -> str:
    """Formats numbers localized to English or Hindi (Devanagari numerals)."""
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        formatted_en = f"{val:,}" if isinstance(val, int) else f"{val:,.2f}"
    else:
        formatted_en = str(val)

    if lang == "hi":
        western_to_devanagari = str.maketrans("0123456789", "०१२३४५६७८९")
        return formatted_en.translate(western_to_devanagari)
    return formatted_en
