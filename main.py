from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pathlib import Path
import os
import json

from dotenv import load_dotenv
from google import genai

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "final_manganese_ml_dataset.csv"

EQUIPMENT_MODEL_PATH = (
    BASE_DIR / "equipment_risk_model.pkl"
)

SHORTFALL_MODEL_PATH = (
    BASE_DIR / "production_shortfall_xgboost_model.json"
)

METADATA_PATH = (
    BASE_DIR / "production_shortfall_model_metadata.pkl"
)

HTML_FILE_PATH = (
    BASE_DIR / "my_gpt.html"
)


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = None

if GEMINI_API_KEY:

    try:
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print("✓ Gemini client initialized successfully")

    except Exception as e:

        print("✗ Gemini initialization error:")
        print(e)

else:

    print("⚠ GEMINI_API_KEY not found")


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="GeoMn Mining Intelligence API",
    description="""
    Hybrid AI/ML based Manganese Mining Risk Prediction System.

    ML Models:
    - Equipment Breakdown Risk Model
    - Production Shortfall XGBoost Model

    Generative AI:
    - Google Gemini for intelligent operational recommendations
    """
)


# =========================================================
# CORS CONFIGURATION
# =========================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)


# =========================================================
# LOAD DATASET
# =========================================================

try:

    df = pd.read_csv(DATA_PATH)

    print("✓ Historical dataset loaded successfully")

    print("Dataset rows:", len(df))

except Exception as e:

    df = None

    print("✗ DATASET ERROR:")

    print(e)


# =========================================================
# LOAD EQUIPMENT RISK MODEL
# =========================================================

try:

    equipment_model = joblib.load(
        EQUIPMENT_MODEL_PATH
    )

    print(
        "✓ Equipment risk model loaded successfully"
    )

except Exception as e:

    equipment_model = None

    print("✗ EQUIPMENT MODEL ERROR:")

    print(e)


# =========================================================
# LOAD PRODUCTION SHORTFALL MODEL
# =========================================================

try:

    shortfall_model = xgb.XGBClassifier()

    shortfall_model.load_model(
        SHORTFALL_MODEL_PATH
    )

    print(
        "✓ Production shortfall XGBoost model loaded successfully"
    )

except Exception as e:

    shortfall_model = None

    print("✗ SHORTFALL MODEL ERROR:")

    print(e)


# =========================================================
# LOAD MODEL METADATA
# =========================================================

try:

    metadata = joblib.load(
        METADATA_PATH
    )

    print(
        "✓ Model metadata loaded successfully"
    )

except Exception as e:

    metadata = None

    print("✗ METADATA ERROR:")

    print(e)


# =========================================================
# REQUEST INPUT MODEL
# =========================================================

class PredictionInput(BaseModel):

    state: str

    district: str

    weather_condition: str

    production_tonnes: float | None = None


# =========================================================
# WEATHER SCENARIO GENERATOR
# =========================================================

def get_weather_values(
    district_data,
    weather_condition
):

    weather_condition = (
        weather_condition
        .lower()
        .strip()
    )


    avg_temperature = float(

        district_data[
            "Avg_Temperature_C"
        ].mean()

    )


    avg_rainfall = float(

        district_data[
            "Total_Rainfall_mm"
        ].mean()

    )


    avg_humidity = float(

        district_data[
            "Avg_Humidity_pct"
        ].mean()

    )


    # GOOD WEATHER
    if weather_condition == "good":

        temperature = float(

            district_data[
                "Avg_Temperature_C"
            ].quantile(0.25)

        )


        rainfall = float(

            district_data[
                "Total_Rainfall_mm"
            ].quantile(0.25)

        )


        humidity = float(

            district_data[
                "Avg_Humidity_pct"
            ].quantile(0.25)

        )


    # BAD WEATHER
    elif weather_condition == "bad":

        temperature = float(

            district_data[
                "Avg_Temperature_C"
            ].quantile(0.75)

        )


        rainfall = float(

            district_data[
                "Total_Rainfall_mm"
            ].quantile(0.75)

        )


        humidity = float(

            district_data[
                "Avg_Humidity_pct"
            ].quantile(0.75)

        )


    # WORST WEATHER
    elif weather_condition == "worst":

        temperature = float(

            district_data[
                "Avg_Temperature_C"
            ].max()

        )


        rainfall = float(

            district_data[
                "Total_Rainfall_mm"
            ].max()

        )


        humidity = float(

            district_data[
                "Avg_Humidity_pct"
            ].max()

        )


    # NORMAL WEATHER
    else:

        temperature = avg_temperature

        rainfall = avg_rainfall

        humidity = avg_humidity


    return (

        temperature,

        rainfall,

        humidity

    )


# =========================================================
# STRESS CALCULATION ENGINE
# =========================================================

def calculate_stress(

    district_data,

    production,

    temperature,

    rainfall,

    humidity

):


    # -------------------------
    # PRODUCTION STRESS
    # -------------------------

    district_avg_production = float(

        district_data[
            "Production_Tonnes"
        ].mean()

    )


    production_stress = (

        district_avg_production
        - production

    ) / district_avg_production


    production_stress = float(

        np.clip(
            production_stress,
            0,
            1
        )

    )


    # -------------------------
    # TEMPERATURE STRESS
    # -------------------------

    temp_mean = float(

        df[
            "Avg_Temperature_C"
        ].mean()

    )


    temp_std = float(

        df[
            "Avg_Temperature_C"
        ].std()

    )


    if temp_std == 0:

        temp_std = 1


    temperature_stress = abs(

        temperature
        - temp_mean

    ) / (

        2 * temp_std

    )


    temperature_stress = float(

        np.clip(
            temperature_stress,
            0,
            1
        )

    )


    # -------------------------
    # RAINFALL STRESS
    # -------------------------

    rain_min = float(

        df[
            "Total_Rainfall_mm"
        ].min()

    )


    rain_max = float(

        df[
            "Total_Rainfall_mm"
        ].max()

    )


    rain_range = rain_max - rain_min


    if rain_range == 0:

        rain_range = 1


    rainfall_stress = (

        rainfall
        - rain_min

    ) / rain_range


    rainfall_stress = float(

        np.clip(
            rainfall_stress,
            0,
            1
        )

    )


    # -------------------------
    # HUMIDITY STRESS
    # -------------------------

    humidity_min = float(

        df[
            "Avg_Humidity_pct"
        ].min()

    )


    humidity_max = float(

        df[
            "Avg_Humidity_pct"
        ].max()

    )


    humidity_range = (

        humidity_max
        - humidity_min

    )


    if humidity_range == 0:

        humidity_range = 1


    humidity_stress = (

        humidity
        - humidity_min

    ) / humidity_range


    humidity_stress = float(

        np.clip(
            humidity_stress,
            0,
            1
        )

    )


    # -------------------------
    # OVERALL WEATHER STRESS
    # -------------------------

    weather_stress = (

        temperature_stress
        + rainfall_stress
        + humidity_stress

    ) / 3


    return {

        "production_stress":
            round(
                production_stress,
                3
            ),

        "temperature_stress":
            round(
                temperature_stress,
                3
            ),

        "rainfall_stress":
            round(
                rainfall_stress,
                3
            ),

        "humidity_stress":
            round(
                humidity_stress,
                3
            ),

        "weather_stress":
            round(
                weather_stress,
                3
            )

    }


# =========================================================
# LOCAL RULE BASED AI FALLBACK
# =========================================================

def generate_local_recommendations(

    production_stress,

    temperature_stress,

    rainfall_stress,

    humidity_stress,

    weather_stress,

    equipment_risk,

    shortfall_risk

):


    recommendations = []

    risk_factors = []


    factors = {

        "Production Performance":
            production_stress,

        "Temperature Conditions":
            temperature_stress,

        "Rainfall Conditions":
            rainfall_stress,

        "Humidity Conditions":
            humidity_stress,

        "Overall Weather Conditions":
            weather_stress,

        "Equipment Reliability":
            equipment_risk / 100

    }


    sorted_factors = sorted(

        factors.items(),

        key=lambda x: x[1],

        reverse=True

    )


    # TOP 3 RISK FACTORS

    for factor, value in sorted_factors[:3]:

        risk_factors.append({

            "factor":
                factor,

            "severity_score":
                round(
                    float(value),
                    3
                )

        })


    # PRODUCTION

    if production_stress >= 0.4:

        recommendations.append({

            "priority":
                "High",

            "issue_detected":
                "Production Performance",

            "recommended_actions": [

                "Identify operational bottlenecks.",

                "Review production scheduling.",

                "Improve transportation and material handling."

            ]

        })


    elif production_stress >= 0.2:

        recommendations.append({

            "priority":
                "Medium",

            "issue_detected":
                "Production Monitoring",

            "recommended_actions": [

                "Monitor daily production targets.",

                "Improve shift planning."

            ]

        })


    # EQUIPMENT

    if equipment_risk >= 20:

        recommendations.append({

            "priority":
                "High",

            "issue_detected":
                "Equipment Reliability",

            "recommended_actions": [

                "Prioritize preventive maintenance.",

                "Inspect critical machines.",

                "Increase equipment monitoring."

            ]

        })


    elif equipment_risk >= 10:

        recommendations.append({

            "priority":
                "Medium",

            "issue_detected":
                "Equipment Monitoring",

            "recommended_actions": [

                "Schedule routine maintenance.",

                "Inspect high-use equipment."

            ]

        })


    # TEMPERATURE

    if temperature_stress >= 0.6:

        recommendations.append({

            "priority":
                "High",

            "issue_detected":
                "Temperature Stress",

            "recommended_actions": [

                "Monitor equipment temperature.",

                "Plan heat-sensitive work carefully.",

                "Ensure adequate cooling."

            ]

        })


    # RAINFALL

    if rainfall_stress >= 0.6:

        recommendations.append({

            "priority":
                "High",

            "issue_detected":
                "Heavy Rainfall Risk",

            "recommended_actions": [

                "Improve mine drainage.",

                "Monitor haul roads.",

                "Prepare alternate operational plans."

            ]

        })


    # HUMIDITY

    if humidity_stress >= 0.6:

        recommendations.append({

            "priority":
                "Medium",

            "issue_detected":
                "High Humidity",

            "recommended_actions": [

                "Protect electrical systems.",

                "Inspect for corrosion.",

                "Increase environmental monitoring."

            ]

        })


    # WEATHER

    if weather_stress >= 0.7:

        recommendations.append({

            "priority":
                "High",

            "issue_detected":
                "Adverse Weather",

            "recommended_actions": [

                "Use weather-based production planning.",

                "Prepare alternative schedules.",

                "Increase operational monitoring."

            ]

        })


    # LOW RISK

    if not recommendations:

        recommendations.append({

            "priority":
                "Low",

            "issue_detected":
                "Stable Operations",

            "recommended_actions": [

                "Continue regular monitoring.",

                "Maintain preventive maintenance.",

                "Track weather changes."

            ]

        })


    top_factor = risk_factors[0]["factor"]


    summary = (

        f"Production shortfall risk is {shortfall_risk}. "

        f"The primary factor requiring attention is "

        f"{top_factor}."

    )


    return {

        "summary":
            summary,

        "top_risk_factors":
            risk_factors,

        "recommendations":
            recommendations

    }


# =========================================================
# GEMINI AI ANALYSIS
# =========================================================

def generate_gemini_analysis(

    state,

    district,

    production,

    production_risk,

    equipment_risk,

    temperature,

    rainfall,

    humidity,

    production_stress,

    weather_stress,

    local_analysis

):


    # FALLBACK IF API NOT AVAILABLE

    if gemini_client is None:

        print(
            "⚠ Gemini unavailable. Using local AI."
        )

        return None


    try:


        prompt = f"""
You are GeoMn AI, an intelligent mining operations assistant.

GeoMn is a hybrid AI/ML system for manganese mining risk analysis.

IMPORTANT:
The numerical predictions below were generated by machine learning models.
Do NOT change or contradict the predicted risk levels.
Your role is to explain the results and provide operational recommendations.

==============================
LOCATION
==============================

State: {state}

District: {district}


==============================
ML PREDICTION RESULTS
==============================

Production Used: {production:.2f} tonnes

Production Shortfall Risk:
{production_risk}

Equipment Breakdown Risk:
{equipment_risk:.2f}%

Temperature:
{temperature:.2f} °C

Rainfall:
{rainfall:.2f} mm

Humidity:
{humidity:.2f}%

Production Stress:
{production_stress * 100:.1f}%

Weather Stress:
{weather_stress * 100:.1f}%


==============================
LOCAL RISK ANALYSIS
==============================

Top Risk Factors:

{json.dumps(local_analysis["top_risk_factors"], indent=2)}


==============================
YOUR TASK
==============================

Provide an intelligent operational analysis.

Use EXACTLY these sections:

1. EXECUTIVE SUMMARY
Briefly explain the overall situation.

2. PRIMARY RISKS
Identify the most important operational risks based ONLY on the ML results.

3. RECOMMENDED ACTIONS
Give 3 to 5 practical actions ranked by priority.

4. OPERATIONAL INSIGHT
Explain what the mining management team should monitor next.

Rules:

- Do not invent geological reserves.
- Do not claim access to real-time mine sensors.
- Do not change ML predictions.
- Keep recommendations practical.
- Be concise and professional.
- Focus on manganese mining operations.
"""


        response = (

            gemini_client
            .models
            .generate_content(

                model="gemini-2.5-flash",

                contents=prompt

            )

        )


        if response and response.text:

            return response.text


        return None


    except Exception as e:

        print(
            "✗ Gemini API Error:"
        )

        print(e)

        return None


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/")

def read_root():

    if not HTML_FILE_PATH.exists():

        raise HTTPException(

            status_code=404,

            detail="Frontend HTML file not found"

        )


    return FileResponse(

        HTML_FILE_PATH,

        media_type="text/html"

    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")

def health():

    return {

        "status":
            "healthy",

        "gemini_available":
            gemini_client is not None,

        "gemini_api_key_configured":
            bool(GEMINI_API_KEY),

        "equipment_model_loaded":
            equipment_model is not None,

        "shortfall_model_loaded":
            shortfall_model is not None,

        "metadata_loaded":
            metadata is not None,

        "dataset_loaded":
            df is not None

    }


# =========================================================
# GET STATES
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

        .unique()

        .tolist()

    )


    return {

        "states":
            states

    }


# =========================================================
# GET DISTRICTS
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

        .str.lower()

        .str.strip()

        ==

        state.lower().strip()

    ]


    if state_data.empty:

        raise HTTPException(

            status_code=404,

            detail="State not found"

        )


    districts = sorted(

        state_data["District"]

        .dropna()

        .unique()

        .tolist()

    )


    return {

        "state":
            state,

        "districts":
            districts

    }


# =========================================================
# MAIN HYBRID PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")

def predict(data: PredictionInput):


    # ----------------------------------
    # SYSTEM CHECK
    # ----------------------------------

    if df is None:

        raise HTTPException(

            status_code=500,

            detail="Dataset not loaded"

        )


    if equipment_model is None:

        raise HTTPException(

            status_code=500,

            detail="Equipment model not loaded"

        )


    if shortfall_model is None:

        raise HTTPException(

            status_code=500,

            detail="Production shortfall model not loaded"

        )


    # ----------------------------------
    # FILTER DISTRICT DATA
    # ----------------------------------

    district_data = df[

        (

            df["State"]

            .astype(str)

            .str.lower()

            .str.strip()

            ==

            data.state.lower().strip()

        )

        &

        (

            df["District"]

            .astype(str)

            .str.lower()

            .str.strip()

            ==

            data.district.lower().strip()

        )

    ].copy()


    if district_data.empty:

        raise HTTPException(

            status_code=404,

            detail="Selected State/District not found"

        )


    # ----------------------------------
    # PRODUCTION INPUT
    # ----------------------------------

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

            "Manual user input"

        )


    else:


        latest_year_data = district_data[

            district_data["Year"]

            .astype(str)

            .str.strip()

            == "2025-26"

        ]


        if not latest_year_data.empty:


            latest_row = (

                latest_year_data.iloc[-1]

            )


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


            latest_row = (

                district_data.iloc[-1]

            )


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


    # ----------------------------------
    # WEATHER SCENARIO
    # ----------------------------------

    temperature, rainfall, humidity = (

        get_weather_values(

            district_data,

            data.weather_condition

        )

    )


    # ----------------------------------
    # STRESS CALCULATION
    # ----------------------------------

    stress = calculate_stress(

        district_data,

        production,

        temperature,

        rainfall,

        humidity

    )


    # ==================================
    # ML MODEL 1
    # EQUIPMENT RISK
    # ==================================

    equipment_features = pd.DataFrame(

        [[

            production,

            temperature,

            rainfall,

            humidity,

            stress["production_stress"],

            stress["temperature_stress"],

            stress["rainfall_stress"],

            stress["humidity_stress"]

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


    equipment_risk = float(

        equipment_model.predict(

            equipment_features

        )[0]

    )


    equipment_risk = float(

        np.clip(

            equipment_risk,

            0,

            100

        )

    )


    # ==================================
    # ML MODEL 2
    # PRODUCTION SHORTFALL
    # ==================================

    shortfall_features = [

        temperature,

        rainfall,

        humidity,

        stress["temperature_stress"],

        stress["rainfall_stress"],

        stress["humidity_stress"],

        stress["weather_stress"],

        stress["production_stress"],

        equipment_risk

    ]


    prediction = (

        shortfall_model.predict(

            np.array([

                shortfall_features

            ])

        )[0]

    )


    prediction = int(prediction)


    # ----------------------------------
    # LABEL MAPPING
    # ----------------------------------

    reverse_label_map = {

        0: "Low",

        1: "Medium",

        2: "High"

    }


    if (

        isinstance(metadata, dict)

        and

        "reverse_label_map" in metadata

    ):

        reverse_label_map = {

            int(k): v

            for k, v in

            metadata[
                "reverse_label_map"
            ].items()

        }


    shortfall_risk = (

        reverse_label_map.get(

            prediction,

            "Unknown"

        )

    )


    # ==================================
    # LOCAL AI ANALYSIS
    # ==================================

    local_analysis = (

        generate_local_recommendations(

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


    # ==================================
    # GEMINI GENERATIVE AI
    # ==================================

    gemini_analysis = (

        generate_gemini_analysis(

            state=data.state,

            district=data.district,

            production=production,

            production_risk=

                shortfall_risk,

            equipment_risk=

                equipment_risk,

            temperature=

                temperature,

            rainfall=

                rainfall,

            humidity=

                humidity,

            production_stress=

                stress[
                    "production_stress"
                ],

            weather_stress=

                stress[
                    "weather_stress"
                ],

            local_analysis=

                local_analysis

        )

    )


    # ==================================
    # HYBRID AI DECISION
    # ==================================

    if gemini_analysis:

        recommendation_source = (

            "Hybrid AI: ML Risk Prediction + Gemini Analysis"

        )

        final_analysis = gemini_analysis


    else:

        recommendation_source = (

            "ML Risk Prediction + Local AI Fallback"

        )

        final_analysis = (

            local_analysis["summary"]

        )


    # ==================================
    # FINAL RESPONSE
    # ==================================

    return {


        # LOCATION

        "state":

            data.state,


        "district":

            data.district,


        # SYSTEM TYPE

        "system_type":

            "Hybrid ML + Generative AI",


        # PRODUCTION

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


        # WEATHER

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


        # STRESS

        "stress_analysis":

            stress,


        # ML OUTPUT

        "ml_predictions": {

            "equipment_breakdown_risk_percent":

                round(
                    equipment_risk,
                    2
                ),

            "production_shortfall_risk":

                shortfall_risk

        },


        # LOCAL AI

        "local_risk_analysis":

            local_analysis,


        # GEMINI AI

        "gemini_analysis":

            gemini_analysis,


        # FINAL HYBRID OUTPUT

        "final_ai_analysis":

            final_analysis,


        "recommendation_source":

            recommendation_source

    }
