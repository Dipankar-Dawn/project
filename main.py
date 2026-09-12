from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import os
from pathlib import Path
import json
import requests
import ee
from dotenv import load_dotenv
from google import genai

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

from sentinel2 import (
    initialize_earth_engine,
    generate_manganese_map
)


# =========================================================
# BASIC SETUP
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="GeoMn Mining Risk API",
    description="AI-powered Manganese Production Risk Platform",
    version="1.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# GEMINI
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = None

if GEMINI_API_KEY:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print("Gemini AI initialized.")

    except Exception as e:

        print(
            "Gemini initialization failed:",
            e
        )

else:

    print(
        "GEMINI_API_KEY not found. "
        "Using offline recommendation engine."
    )


# =========================================================
# EARTH ENGINE
# =========================================================

EARTH_ENGINE_AVAILABLE = False

try:

    initialize_earth_engine()

    EARTH_ENGINE_AVAILABLE = True

    print(
        "Google Earth Engine initialized."
    )

except Exception as e:

    print(
        "Earth Engine unavailable:",
        e
    )

    EARTH_ENGINE_AVAILABLE = False


# =========================================================
# DATA / MODEL PATHS
# =========================================================

DATA_PATH = (
    BASE_DIR /
    "data" /
    "final_manganese_ml_dataset.csv"
)

EQUIPMENT_MODEL_PATH = (
    BASE_DIR /
    "equipment_risk_model.pkl"
)

SHORTFALL_MODEL_PATH = (
    BASE_DIR /
    "production_shortfall_xgboost_model.json"
)

METADATA_PATH = (
    BASE_DIR /
    "production_shortfall_model_metadata.pkl"
)


# =========================================================
# LOAD DATASET
# =========================================================

df = None

try:

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        "Dataset loaded:",
        df.shape
    )

except Exception as e:

    print(
        "Dataset loading failed:",
        e
    )


# =========================================================
# LOAD EQUIPMENT MODEL
# =========================================================

equipment_model = None

try:

    equipment_model = joblib.load(
        EQUIPMENT_MODEL_PATH
    )

    print(
        "Equipment model loaded."
    )

except Exception as e:

    print(
        "Equipment model loading failed:",
        e
    )


# =========================================================
# LOAD XGBOOST SHORTFALL MODEL
# =========================================================

shortfall_model = None

try:

    shortfall_model = xgb.XGBClassifier()

    shortfall_model.load_model(
        SHORTFALL_MODEL_PATH
    )

    print(
        "XGBoost shortfall model loaded."
    )

except Exception as e:

    print(
        "Shortfall model loading failed:",
        e
    )

    shortfall_model = None


# =========================================================
# LOAD MODEL METADATA
# =========================================================

metadata = None

try:

    metadata = joblib.load(
        METADATA_PATH
    )

    print(
        "Model metadata loaded."
    )

except Exception as e:

    print(
        "Metadata loading failed:",
        e
    )


# =========================================================
# REQUEST MODELS
# =========================================================

class PredictionInput(BaseModel):

    state: str

    district: str

    weather_condition: str

    equipment_mode: str = "Auto"

    production_tonnes: float | None = None


class MapRequest(BaseModel):

    west: float

    south: float

    east: float

    north: float


# =========================================================
# ROOT PAGE
# =========================================================

@app.get("/")
def read_root():

    html_path = (
        BASE_DIR /
        "my_gpt.html"
    )

    if not html_path.exists():

        raise HTTPException(
            status_code=404,
            detail="my_gpt.html not found"
        )

    return FileResponse(
        html_path,
        media_type="text/html"
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {

        "status": "online",

        "service":
            "GeoMn Mining Risk API",

        "dataset_loaded":
            df is not None,

        "equipment_model_loaded":
            equipment_model is not None,

        "shortfall_model_loaded":
            shortfall_model is not None,

        "metadata_loaded":
            metadata is not None,

        "earth_engine_available":
            EARTH_ENGINE_AVAILABLE,

        "gemini_available":
            gemini_client is not None

    }


# =========================================================
# STATES
# =========================================================

@app.get("/states")
def get_states():

    if df is None:

        raise HTTPException(
            status_code=500,
            detail="Dataset not loaded"
        )

    states = sorted(
        df["State"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return {

        "states": states

    }


# =========================================================
# DISTRICTS
# =========================================================

@app.get("/districts/{state}")
def get_districts(state: str):

    if df is None:

        raise HTTPException(
            status_code=500,
            detail="Dataset not loaded"
        )

    state_data = df[
        df["State"]
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        state.strip().lower()
    ]

    if state_data.empty:

        raise HTTPException(
            status_code=404,
            detail="State not found"
        )

    districts = sorted(
        state_data["District"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return {

        "state": state,

        "districts": districts

    }


# =========================================================
# WEATHER
# =========================================================

def get_weather_values(
    district_data,
    weather_condition,
    state,
    district
):

    condition = (
        weather_condition
        .lower()
        .strip()
    )


    # -----------------------------------------------------
    # LIVE / AUTO WEATHER
    # -----------------------------------------------------

    if condition == "auto":

        try:

            geocode_url = (
                "https://geocoding-api.open-meteo.com/"
                "v1/search"
            )

            geo_response = requests.get(
                geocode_url,
                params={
                    "name": district,
                    "count": 1,
                    "language": "en",
                    "format": "json"
                },
                timeout=4
            )

            geo_data = (
                geo_response.json()
            )

            if geo_data.get("results"):

                location =
                    geo_data["results"][0]

                latitude = (
                    location["latitude"]
                )

                longitude = (
                    location["longitude"]
                )

                weather_url = (
                    "https://api.open-meteo.com/"
                    "v1/forecast"
                )

                weather_response = requests.get(
                    weather_url,
                    params={
                        "latitude": latitude,
                        "longitude": longitude,
                        "current": (
                            "temperature_2m,"
                            "relative_humidity_2m,"
                            "precipitation"
                        )
                    },
                    timeout=4
                )

                weather_data = (
                    weather_response.json()
                )

                current = (
                    weather_data
                    .get("current", {})
                )

                temperature = float(
                    current.get(
                        "temperature_2m",
                        25
                    )
                )

                humidity = float(
                    current.get(
                        "relative_humidity_2m",
                        60
                    )
                )

                rainfall = float(
                    current.get(
                        "precipitation",
                        0
                    )
                )

                return (
                    temperature,
                    rainfall,
                    humidity
                )

        except Exception as e:

            print(
                "Live weather unavailable:",
                e
            )


        # -------------------------------------------------
        # OFFLINE FALLBACK
        # -------------------------------------------------

        return get_historical_weather(
            district_data
        )


    # =====================================================
    # MANUAL WEATHER SCENARIOS
    # =====================================================

    historical = get_historical_weather(
        district_data
    )

    base_temperature = historical[0]
    base_rainfall = historical[1]
    base_humidity = historical[2]


    if condition == "good":

        return (
            base_temperature * 0.95,
            base_rainfall * 0.60,
            base_humidity * 0.90
        )


    elif condition == "normal":

        return (
            base_temperature,
            base_rainfall,
            base_humidity
        )


    elif condition == "bad":

        return (
            base_temperature * 1.08,
            base_rainfall * 1.40,
            min(
                100,
                base_humidity * 1.08
            )
        )


    elif condition == "worst":

        return (
            base_temperature * 1.15,
            base_rainfall * 1.80,
            min(
                100,
                base_humidity * 1.15
            )
        )


    return historical


# =========================================================
# HISTORICAL WEATHER
# =========================================================

def get_historical_weather(
    district_data
):

    temperature_columns = [
        "Avg_Temperature_C",
        "Temperature_C",
        "temperature"
    ]

    rainfall_columns = [
        "Total_Rainfall_mm",
        "Rainfall_mm",
        "rainfall"
    ]

    humidity_columns = [
        "Avg_Humidity_pct",
        "Humidity_pct",
        "humidity"
    ]


    temperature = 25.0

    rainfall = 100.0

    humidity = 60.0


    for column in temperature_columns:

        if column in district_data.columns:

            values = pd.to_numeric(
                district_data[column],
                errors="coerce"
            ).dropna()

            if not values.empty:

                temperature = float(
                    values.mean()
                )

                break


    for column in rainfall_columns:

        if column in district_data.columns:

            values = pd.to_numeric(
                district_data[column],
                errors="coerce"
            ).dropna()

            if not values.empty:

                rainfall = float(
                    values.mean()
                )

                break


    for column in humidity_columns:

        if column in district_data.columns:

            values = pd.to_numeric(
                district_data[column],
                errors="coerce"
            ).dropna()

            if not values.empty:

                humidity = float(
                    values.mean()
                )

                break


    return (
        temperature,
        rainfall,
        humidity
    )


# =========================================================
# STRESS CALCULATION
# =========================================================

def calculate_stress(
    district_data,
    production,
    temperature,
    rainfall,
    humidity
):

    # -----------------------------------------------------
    # PRODUCTION STRESS
    # -----------------------------------------------------

    historical_production = pd.to_numeric(
        district_data[
            "Production_Tonnes"
        ],
        errors="coerce"
    ).dropna()


    if historical_production.empty:

        production_stress = 0.0

    else:

        historical_avg = float(
            historical_production.mean()
        )

        if historical_avg <= 0:

            production_stress = 0.0

        else:

            production_difference = (
                historical_avg - production
            ) / historical_avg

            production_stress = float(
                np.clip(
                    production_difference,
                    0,
                    1
                )
            )


    # -----------------------------------------------------
    # TEMPERATURE STRESS
    # -----------------------------------------------------

    temperature_stress = float(
        np.clip(
            abs(temperature - 25) / 20,
            0,
            1
        )
    )


    # -----------------------------------------------------
    # RAINFALL STRESS
    # -----------------------------------------------------

    rainfall_stress = float(
        np.clip(
            rainfall / 500,
            0,
            1
        )
    )


    # -----------------------------------------------------
    # HUMIDITY STRESS
    # -----------------------------------------------------

    humidity_stress = float(
        np.clip(
            abs(humidity - 60) / 40,
            0,
            1
        )
    )


    # -----------------------------------------------------
    # OVERALL WEATHER STRESS
    # -----------------------------------------------------

    weather_stress = float(
        np.mean([
            temperature_stress,
            rainfall_stress,
            humidity_stress
        ])
    )


    return {

        "production_stress":
            round(
                production_stress,
                4
            ),

        "temperature_stress":
            round(
                temperature_stress,
                4
            ),

        "rainfall_stress":
            round(
                rainfall_stress,
                4
            ),

        "humidity_stress":
            round(
                humidity_stress,
                4
            ),

        "weather_stress":
            round(
                weather_stress,
                4
            )

    }


# =========================================================
# LOCAL RECOMMENDATION ENGINE
# =========================================================

def generate_ai_recommendations(
    production_stress,
    temperature_stress,
    rainfall_stress,
    humidity_stress,
    weather_stress,
    equipment_risk,
    shortfall_risk
):

    factors = [

        (
            "Production Stress",
            production_stress
        ),

        (
            "Temperature Stress",
            temperature_stress
        ),

        (
            "Rainfall Stress",
            rainfall_stress
        ),

        (
            "Humidity Stress",
            humidity_stress
        ),

        (
            "Weather Stress",
            weather_stress
        ),

        (
            "Equipment Breakdown Risk",
            equipment_risk / 100
        )

    ]


    factors.sort(
        key=lambda x: x[1],
        reverse=True
    )


    top_factors = [

        {
            "factor": name,
            "severity_score":
                round(
                    score,
                    4
                )
        }

        for name, score
        in factors[:3]

    ]


    recommendations = []


    # =====================================================
    # PRODUCTION
    # =====================================================

    if production_stress >= 0.60:

        recommendations.append({

            "priority": "Critical",

            "title":
                "Production Stress Mitigation",

            "issue_detected":
                "Production stress is high.",

            "why_it_matters":
                "The current production level is significantly below the historical baseline.",

            "recommended_actions": [

                "Review current mining and production operations.",

                "Identify production bottlenecks.",

                "Prioritize high-yield mining zones.",

                "Monitor production against historical targets."

            ]

        })

    elif production_stress >= 0.30:

        recommendations.append({

            "priority": "High",

            "title":
                "Production Monitoring",

            "issue_detected":
                "Moderate production stress detected.",

            "why_it_matters":
                "Production is showing deviation from the historical baseline.",

            "recommended_actions": [

                "Monitor daily production.",

                "Review operational bottlenecks.",

                "Compare production with historical performance."

            ]

        })


    # =====================================================
    # EQUIPMENT
    # =====================================================

    if equipment_risk >= 70:

        recommendations.append({

            "priority": "Critical",

            "title":
                "Equipment Reliability",

            "issue_detected":
                "High equipment breakdown risk.",

            "why_it_matters":
                "Equipment failure can directly reduce mining and processing capacity.",

            "recommended_actions": [

                "Perform preventive maintenance.",

                "Inspect critical mining equipment.",

                "Keep essential spare parts available.",

                "Prioritize high-risk equipment."

            ]

        })

    elif equipment_risk >= 40:

        recommendations.append({

            "priority": "High",

            "title":
                "Equipment Monitoring",

            "issue_detected":
                "Moderate equipment breakdown risk.",

            "why_it_matters":
                "Unexpected equipment downtime can contribute to production shortfall.",

            "recommended_actions": [

                "Increase equipment inspection frequency.",

                "Review maintenance schedules.",

                "Monitor critical machinery."

            ]

        })


    # =====================================================
    # WEATHER
    # =====================================================

    if weather_stress >= 0.60:

        recommendations.append({

            "priority": "High",

            "title":
                "Weather Risk Management",

            "issue_detected":
                "High environmental stress.",

            "why_it_matters":
                "Adverse environmental conditions may affect mining operations and productivity.",

            "recommended_actions": [

                "Monitor weather conditions continuously.",

                "Adjust mining schedules when required.",

                "Prepare drainage and site safety measures.",

                "Protect critical operational infrastructure."

            ]

        })

    elif weather_stress >= 0.30:

        recommendations.append({

            "priority": "Medium",

            "title":
                "Weather Monitoring",

            "issue_detected":
                "Moderate weather stress.",

            "why_it_matters":
                "Changing environmental conditions can influence mining operations.",

            "recommended_actions": [

                "Continue weather monitoring.",

                "Review operational schedules.",

                "Maintain site preparedness."

            ]

        })


    if not recommendations:

        recommendations.append({

            "priority": "Low",

            "title":
                "Normal Operations",

            "issue_detected":
                "No major risk factor detected.",

            "why_it_matters":
                "Current conditions do not indicate significant operational stress.",

            "recommended_actions": [

                "Continue normal operations.",

                "Maintain routine equipment inspections.",

                "Continue production monitoring."

            ]

        })


    return {

        "top_risk_factors":
            top_factors,

        "recommendations":
            recommendations

    }


# =========================================================
# GEMINI RECOMMENDATION
# =========================================================

def generate_gemini_recommendation(
    state,
    district,
    production_risk,
    equipment_risk,
    weather_stress,
    production_stress,
    trend
):

    if gemini_client is None:

        return None


    try:

        prompt = f"""

You are an industrial mining risk analyst.

Analyze the following manganese mining situation:

State: {state}

District: {district}

Production Shortfall Risk:
{production_risk}

Equipment Breakdown Risk:
{equipment_risk}%

Weather Stress:
{weather_stress}

Production Stress:
{production_stress}

Production Data Source:
{trend}

Provide a concise operational recommendation.

Mention:

1. Main risk
2. Why it matters
3. Immediate actions
4. Monitoring actions

Keep the answer practical for a manganese mining operation.
"""


        response = gemini_client.models.generate_content(

            model="gemini-2.5-flash",

            contents=prompt

        )


        if response and response.text:

            return response.text.strip()


    except Exception as e:

        print(
            "Gemini recommendation failed:",
            e
        )


    return None


# =========================================================
# GENERATE SENTINEL-2 MANGANESE MAP
# =========================================================

@app.post("/generate-map")
def generate_map(
    data: MapRequest
):

    if not EARTH_ENGINE_AVAILABLE:

        raise HTTPException(
            status_code=503,
            detail=(
                "Google Earth Engine "
                "is not available"
            )
        )


    result = generate_manganese_map(

        west=data.west,

        south=data.south,

        east=data.east,

        north=data.north,

        output_file=str(
            BASE_DIR /
            "moil_manganese_map.html"
        )

    )


    if not result.get("success"):

        raise HTTPException(

            status_code=500,

            detail=result.get(
                "error",
                "Sentinel map generation failed"
            )

        )


    return {

        "status": "success",

        "message":
            "Sentinel-2 manganese analysis completed",

        "threshold":
            result.get("threshold"),

        "map_url":
            "/map"

    }


# =========================================================
# SERVE GENERATED SENTINEL MAP
# =========================================================

@app.get("/map")
def get_map():

    map_path = (
        BASE_DIR /
        "moil_manganese_map.html"
    )


    if not map_path.exists():

        raise HTTPException(

            status_code=404,

            detail=
                "Map has not been generated yet"

        )


    return FileResponse(

        map_path,

        media_type="text/html"

    )


# =========================================================
# MAIN PREDICTION
# =========================================================

@app.post("/predict")
def predict(
    data: PredictionInput
):

    if equipment_model is None:

        raise HTTPException(

            status_code=500,

            detail=
                "Equipment model not loaded"

        )


    if shortfall_model is None:

        raise HTTPException(

            status_code=500,

            detail=
                "Shortfall model not loaded"

        )


    if metadata is None:

        raise HTTPException(

            status_code=500,

            detail=
                "Metadata not loaded"

        )


    if df is None:

        raise HTTPException(

            status_code=500,

            detail=
                "Dataset not loaded"

        )


    # =====================================================
    # FIND DISTRICT
    # =====================================================

    district_data = df[

        (

            df["State"]

            .astype(str)

            .str.lower()

            .str.strip()

            ==

            data.state
            .lower()
            .strip()

        )

        &

        (

            df["District"]

            .astype(str)

            .str.lower()

            .str.strip()

            ==

            data.district
            .lower()
            .strip()

        )

    ].copy()


    if district_data.empty:

        raise HTTPException(

            status_code=404,

            detail=
                "Selected State/District not found"

        )


    # =====================================================
    # PRODUCTION
    # =====================================================

    production_source = ""

    selected_year = None


    if (

        data.production_tonnes is not None

        and

        data.production_tonnes > 0

    ):

        production = float(
            data.production_tonnes
        )

        production_source = (
            "Manual input"
        )


    else:

        latest_year_data = district_data[

            district_data["Year"]

            .astype(str)

            .str.strip()

            ==

            "2025-26"

        ]


        if not latest_year_data.empty:

            latest_row =
                latest_year_data.iloc[-1]

            production = float(

                latest_row[
                    "Production_Tonnes"
                ]

            )

            selected_year = "2025-26"

            production_source = (
                "Automatic latest historical data"
            )


        else:

            latest_row =
                district_data.iloc[-1]

            production = float(

                latest_row[
                    "Production_Tonnes"
                ]

            )

            selected_year = str(

                latest_row["Year"]

            )

            production_source = (
                "Latest available historical data"
            )


    # =====================================================
    # WEATHER
    # =====================================================

    temperature, rainfall, humidity = (
        get_weather_values(

            district_data,

            data.weather_condition,

            data.state,

            data.district

        )
    )


    # =====================================================
    # STRESS
    # =====================================================

    stress = calculate_stress(

        district_data,

        production,

        temperature,

        rainfall,

        humidity

    )


    # =====================================================
    # EQUIPMENT FEATURES
    # =====================================================

    equipment_features = pd.DataFrame(

        [[

            production,

            temperature,

            rainfall,

            humidity,

            stress[
                "production_stress"
            ],

            stress[
                "temperature_stress"
            ],

            stress[
                "rainfall_stress"
            ],

            stress[
                "humidity_stress"
            ]

        ]],

        columns=[

            "Production_Tonnes",

            "Avg_Temperature_C",

            "Total_Rainfall_mm",

            "Avg_Humidity_pct",

            "Production_Stress",

            "Temperature_Stress",

            "Rainfall_Stress",

            "Humidity_Stress"

        ]

    )


    # =====================================================
    # EQUIPMENT RISK
    # =====================================================

    if data.equipment_mode == "Auto":

        equipment_risk = float(

            equipment_model.predict(

                equipment_features

            )[0]

        )

        equipment_risk_source = (
            "ML Prediction"
        )


    elif data.equipment_mode == "Low":

        equipment_risk = 10.0

        equipment_risk_source = (
            "Simulation - Low"
        )


    elif data.equipment_mode == "Medium":

        equipment_risk = 50.0

        equipment_risk_source = (
            "Simulation - Medium"
        )


    elif data.equipment_mode == "High":

        equipment_risk = 85.0

        equipment_risk_source = (
            "Simulation - High"
        )


    else:

        raise HTTPException(

            status_code=400,

            detail=
                "Invalid equipment mode"

        )


    equipment_risk = float(

        np.clip(

            equipment_risk,

            0,

            100

        )

    )


    # =====================================================
    # SHORTFALL MODEL
    # =====================================================

    shortfall_features = [

        temperature,

        rainfall,

        humidity,

        stress[
            "temperature_stress"
        ],

        stress[
            "rainfall_stress"
        ],

        stress[
            "humidity_stress"
        ],

        stress[
            "weather_stress"
        ],

        stress[
            "production_stress"
        ],

        equipment_risk

    ]


    prediction = shortfall_model.predict(

        np.array(
            [shortfall_features]
        )

    )[0]


    prediction = int(
        prediction
    )


    # =====================================================
    # LABEL MAP
    # =====================================================

    reverse_label_map = {

        0: "Low",

        1: "Medium",

        2: "High"

    }


    if (

        isinstance(
            metadata,
            dict
        )

        and

        "reverse_label_map"
        in metadata

    ):

        reverse_label_map = {

            int(k): v

            for k, v
            in metadata[
                "reverse_label_map"
            ].items()

        }


    shortfall_risk = (

        reverse_label_map.get(

            prediction,

            "Unknown"

        )

    )


    # =====================================================
    # LOCAL RECOMMENDATIONS
    # =====================================================

    ai_recommendations = (
        generate_ai_recommendations(

            production_stress=
                stress[
                    "production_stress"
                ],

            temperature_stress=
                stress[
                    "temperature_stress"
                ],

            rainfall_stress=
                stress[
                    "rainfall_stress"
                ],

            humidity_stress=
                stress[
                    "humidity_stress"
                ],

            weather_stress=
                stress[
                    "weather_stress"
                ],

            equipment_risk=
                equipment_risk,

            shortfall_risk=
                shortfall_risk

        )
    )


    # =====================================================
    # GEMINI
    # =====================================================

    ai_recommendation = (
        generate_gemini_recommendation(

            state=data.state,

            district=data.district,

            production_risk=
                shortfall_risk,

            equipment_risk=
                round(
                    equipment_risk,
                    2
                ),

            weather_stress=
                round(
                    stress[
                        "weather_stress"
                    ],
                    2
                ),

            production_stress=
                round(
                    stress[
                        "production_stress"
                    ],
                    2
                ),

            trend=production_source

        )
    )


    if ai_recommendation:

        recommendation_source = (
            "Gemini AI Recommendation"
        )

    else:

        recommendation_source = (
            "GeoMn Rule-Based Recommendation "
            "(Offline Fallback)"
        )


    # =====================================================
    # RESPONSE
    # =====================================================

    return {

        "state":
            data.state,

        "district":
            data.district,


        "production_used": {

            "production_tonnes":
                round(
                    production,
                    2
                ),

            "year":
                selected_year,

            "source":
                production_source

        },


        "weather_condition":
            data.weather_condition,


        "weather_values": {

            "temperature_c":
                round(
                    temperature,
                    2
                ),

            "rainfall_mm":
                round(
                    rainfall,
                    2
                ),

            "humidity_percent":
                round(
                    humidity,
                    2
                )

        },


        "stress_analysis":
            stress,


        "equipment_breakdown_risk_percent":
            round(
                equipment_risk,
                2
            ),


        "equipment_risk_source":
            equipment_risk_source,


        "production_shortfall_risk":
            shortfall_risk,


        "ai_recommendations":
            ai_recommendations,


        "ai_recommendation":
            ai_recommendation,


        "recommendation_source":
            recommendation_source

    }


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=int(
            os.getenv(
                "PORT",
                8000
            )
        )

    )
