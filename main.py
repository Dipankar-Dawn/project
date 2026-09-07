from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)

print("Gemini API key found:", bool(GEMINI_API_KEY))
print("Gemini client initialized:", gemini_client is not None)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="GeoMn Mining Risk API",
    description="AI/ML based Manganese Mining Risk Prediction System"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# FILE PATHS
# =========================================================

DATA_PATH = (
    BASE_DIR
    / "data"
    / "final_manganese_ml_dataset.csv"
)

EQUIPMENT_MODEL_PATH = (
    BASE_DIR
    / "equipment_risk_model.pkl"
)

SHORTFALL_MODEL_PATH = (
    BASE_DIR
    / "production_shortfall_xgboost_model.json"
)

METADATA_PATH = (
    BASE_DIR
    / "production_shortfall_model_metadata.pkl"
)

HTML_FILE_PATH = (
    BASE_DIR
    / "my_gpt.html"
)


# =========================================================
# LOAD DATASET
# =========================================================

try:
    df = pd.read_csv(DATA_PATH)
    print("✓ Historical dataset loaded successfully")

except Exception as e:
    df = None
    print("✗ DATASET ERROR:")
    print(e)


# =========================================================
# LOAD EQUIPMENT MODEL
# =========================================================

try:
    equipment_model = joblib.load(EQUIPMENT_MODEL_PATH)
    print("✓ Equipment risk model loaded successfully")

except Exception as e:
    equipment_model = None
    print("✗ EQUIPMENT MODEL ERROR:")
    print(e)


# =========================================================
# LOAD SHORTFALL XGBOOST MODEL
# =========================================================

try:
    shortfall_model = xgb.XGBClassifier()
    shortfall_model.load_model(SHORTFALL_MODEL_PATH)

    print("✓ Production shortfall XGBoost model loaded successfully")

except Exception as e:
    shortfall_model = None
    print("✗ SHORTFALL MODEL ERROR:")
    print(e)


# =========================================================
# LOAD MODEL METADATA
# =========================================================

try:
    metadata = joblib.load(METADATA_PATH)
    print("✓ Model metadata loaded successfully")

except Exception as e:
    metadata = None
    print("✗ METADATA ERROR:")
    print(e)


# =========================================================
# REQUEST MODEL
# =========================================================

class PredictionInput(BaseModel):
    state: str
    district: str
    weather_condition: str
    production_tonnes: float | None = None


# =========================================================
# WEATHER SCENARIO FUNCTION
# =========================================================

def get_weather_values(district_data, weather_condition):

    weather_condition = weather_condition.lower().strip()

    avg_temperature = float(
        district_data["Avg_Temperature_C"].mean()
    )

    avg_rainfall = float(
        district_data["Total_Rainfall_mm"].mean()
    )

    avg_humidity = float(
        district_data["Avg_Humidity_pct"].mean()
    )


    if weather_condition == "good":

        temperature = float(
            district_data["Avg_Temperature_C"].quantile(0.25)
        )

        rainfall = float(
            district_data["Total_Rainfall_mm"].quantile(0.25)
        )

        humidity = float(
            district_data["Avg_Humidity_pct"].quantile(0.25)
        )


    elif weather_condition == "bad":

        temperature = float(
            district_data["Avg_Temperature_C"].quantile(0.75)
        )

        rainfall = float(
            district_data["Total_Rainfall_mm"].quantile(0.75)
        )

        humidity = float(
            district_data["Avg_Humidity_pct"].quantile(0.75)
        )


    elif weather_condition == "worst":

        temperature = float(
            district_data["Avg_Temperature_C"].max()
        )

        rainfall = float(
            district_data["Total_Rainfall_mm"].max()
        )

        humidity = float(
            district_data["Avg_Humidity_pct"].max()
        )


    else:

        temperature = avg_temperature
        rainfall = avg_rainfall
        humidity = avg_humidity


    return temperature, rainfall, humidity


# =========================================================
# CALCULATE STRESS
# =========================================================

def calculate_stress(
    district_data,
    production,
    temperature,
    rainfall,
    humidity
):

    district_avg_production = float(
        district_data["Production_Tonnes"].mean()
    )

    production_stress = (
        district_avg_production - production
    ) / district_avg_production

    production_stress = float(
        np.clip(production_stress, 0, 1)
    )


    temp_mean = float(
        df["Avg_Temperature_C"].mean()
    )

    temp_std = float(
        df["Avg_Temperature_C"].std()
    )

    temperature_stress = abs(
        temperature - temp_mean
    ) / (2 * temp_std)

    temperature_stress = float(
        np.clip(temperature_stress, 0, 1)
    )


    rain_min = float(
        df["Total_Rainfall_mm"].min()
    )

    rain_max = float(
        df["Total_Rainfall_mm"].max()
    )

    rainfall_stress = (
        rainfall - rain_min
    ) / (rain_max - rain_min)

    rainfall_stress = float(
        np.clip(rainfall_stress, 0, 1)
    )


    humidity_min = float(
        df["Avg_Humidity_pct"].min()
    )

    humidity_max = float(
        df["Avg_Humidity_pct"].max()
    )

    humidity_stress = (
        humidity - humidity_min
    ) / (humidity_max - humidity_min)

    humidity_stress = float(
        np.clip(humidity_stress, 0, 1)
    )


    weather_stress = (
        temperature_stress
        + rainfall_stress
        + humidity_stress
    ) / 3


    return {

        "production_stress":
            round(production_stress, 3),

        "temperature_stress":
            round(temperature_stress, 3),

        "rainfall_stress":
            round(rainfall_stress, 3),

        "humidity_stress":
            round(humidity_stress, 3),

        "weather_stress":
            round(weather_stress, 3)
    }


# =========================================================
# LOCAL AI RECOMMENDATION ENGINE
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

    recommendations = []
    risk_factors = []


    factors = {
        "Production Performance": production_stress,
        "Temperature Conditions": temperature_stress,
        "Rainfall Conditions": rainfall_stress,
        "Humidity Conditions": humidity_stress,
        "Overall Weather Conditions": weather_stress,
        "Equipment Reliability": equipment_risk / 100
    }


    sorted_factors = sorted(
        factors.items(),
        key=lambda x: x[1],
        reverse=True
    )


    for factor, value in sorted_factors[:3]:

        risk_factors.append({

            "factor": factor,

            "severity_score":
                round(float(value), 3)
        })


    if production_stress >= 0.4:

        recommendations.append({

            "priority": "High",

            "issue_detected":
                "Production Performance",

            "why_it_matters":
                "Production is significantly below the expected operational level.",

            "recommended_actions": [

                "Identify production delays and bottlenecks.",

                "Improve production scheduling.",

                "Improve transportation and material handling."
            ]
        })


    elif production_stress >= 0.2:

        recommendations.append({

            "priority": "Medium",

            "issue_detected":
                "Production Performance",

            "why_it_matters":
                "Production performance is slightly below the expected level.",

            "recommended_actions": [

                "Monitor daily production targets.",

                "Improve shift planning and machine utilization."
            ]
        })


    if equipment_risk >= 20:

        recommendations.append({

            "priority": "High",

            "issue_detected":
                "Equipment Reliability",

            "why_it_matters":
                "The system detected an increased chance of equipment-related disruption.",

            "recommended_actions": [

                "Prioritize preventive maintenance.",

                "Inspect critical machines before operations.",

                "Monitor equipment performance regularly."
            ]
        })


    elif equipment_risk >= 10:

        recommendations.append({

            "priority": "Medium",

            "issue_detected":
                "Equipment Monitoring",

            "why_it_matters":
                "Some equipment may require additional monitoring.",

            "recommended_actions": [

                "Schedule routine maintenance.",

                "Inspect critical machines regularly."
            ]
        })


    if temperature_stress >= 0.6:

        recommendations.append({

            "priority": "High",

            "issue_detected":
                "High Temperature Stress",

            "why_it_matters":
                "High temperature may affect equipment performance and mining operations.",

            "recommended_actions": [

                "Monitor equipment temperature.",

                "Schedule heat-sensitive work during suitable hours.",

                "Ensure proper cooling and thermal protection."
            ]
        })


    if rainfall_stress >= 0.6:

        recommendations.append({

            "priority": "High",

            "issue_detected":
                "Heavy Rainfall Risk",

            "why_it_matters":
                "Heavy rainfall may affect transportation and mining operations.",

            "recommended_actions": [

                "Improve mine drainage.",

                "Monitor haul roads regularly.",

                "Prepare backup plans for heavy rainfall."
            ]
        })


    if humidity_stress >= 0.6:

        recommendations.append({

            "priority": "Medium",

            "issue_detected":
                "High Humidity",

            "why_it_matters":
                "High humidity may affect equipment performance.",

            "recommended_actions": [

                "Protect electrical equipment from moisture.",

                "Inspect machines for corrosion.",

                "Increase environmental monitoring."
            ]
        })


    if weather_stress >= 0.7:

        recommendations.append({

            "priority": "High",

            "issue_detected":
                "Adverse Weather Conditions",

            "why_it_matters":
                "Combined weather conditions may create operational challenges.",

            "recommended_actions": [

                "Use weather-based operational planning.",

                "Prepare alternative production schedules.",

                "Increase monitoring during adverse weather."
            ]
        })


    if shortfall_risk == "Low":

        recommendations.append({

            "priority": "Low",

            "issue_detected":
                "Current Operational Status",

            "why_it_matters":
                "The overall production shortfall risk is currently low.",

            "recommended_actions": [

                "Continue regular monitoring.",

                "Maintain preventive maintenance.",

                "Track production and weather changes."
            ]
        })


    top_factor = risk_factors[0]["factor"]


    if shortfall_risk == "High":

        ai_summary = (
            f"The system predicts a HIGH production shortfall risk. "
            f"The main area requiring attention is {top_factor}."
        )


    elif shortfall_risk == "Medium":

        ai_summary = (
            f"The system predicts a MODERATE production shortfall risk. "
            f"The main concern is {top_factor}."
        )


    else:

        ai_summary = (
            f"The overall production shortfall risk is currently LOW. "
            f"However, {top_factor} requires regular monitoring."
        )


    return {

        "current_risk": shortfall_risk,

        "ai_summary": ai_summary,

        "top_risk_factors": risk_factors,

        "recommendations": recommendations
    }


# =========================================================
# GEMINI AI RECOMMENDATION
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

        print("Gemini client unavailable - using local fallback")

        return None


    try:

        prompt = f"""
You are an AI mining operations assistant for GeoMn,
a manganese mining intelligence system.

Analyze this mining risk assessment.

Location:
State: {state}
District: {district}

Risk Analysis:
Production Shortfall Risk: {production_risk}
Equipment Breakdown Risk: {equipment_risk}%
Weather Stress: {weather_stress}%
Production Stress: {production_stress}%
Production Data Source: {trend}

Instructions:
1. Identify the main operational concern.
2. Give 3 to 5 practical recommendations.
3. Prioritize the highest risks.
4. Do not invent geological facts.
5. Keep the response concise and professional.
6. Focus on actionable mining operation planning.

Format clearly with headings and bullet points.
"""


        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )


        return response.text


    except Exception as e:

        print(f"Gemini recommendation error: {e}")

        return None


# =========================================================
# HOME
# =========================================================

@app.get("/")
def read_root():

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

        "status": "healthy",

        "gemini_available":
            gemini_client is not None,

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

    return {"states": states}


# =========================================================
# GET DISTRICTS
# =========================================================

@app.get("/districts/{state}")
def get_districts(state: str):

    state_data = df[

        df["State"]
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

        "state": state,

        "districts": districts
    }


# =========================================================
# MAIN PREDICTION
# =========================================================

@app.post("/predict")
def predict(data: PredictionInput):


    if equipment_model is None:
        raise HTTPException(
            status_code=500,
            detail="Equipment model not loaded"
        )


    if shortfall_model is None:
        raise HTTPException(
            status_code=500,
            detail="Shortfall model not loaded"
        )


    if metadata is None:
        raise HTTPException(
            status_code=500,
            detail="Metadata not loaded"
        )


    if df is None:
        raise HTTPException(
            status_code=500,
            detail="Dataset not loaded"
        )


    district_data = df[

        (
            df["State"]
            .str.lower()
            .str.strip()

            ==

            data.state.lower().strip()
        )

        &

        (
            df["District"]
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


    production_source = ""
    selected_year = None


    if (
        data.production_tonnes is not None
        and data.production_tonnes > 0
    ):

        production = float(
            data.production_tonnes
        )

        production_source = "Manual input"


    else:

        latest_year_data = district_data[

            district_data["Year"]
            .astype(str)
            .str.strip()

            == "2025-26"
        ]


        if not latest_year_data.empty:

            latest_row = latest_year_data.iloc[-1]

            production = float(
                latest_row["Production_Tonnes"]
            )

            selected_year = "2025-26"

            production_source = (
                "Automatic latest historical data"
            )


        else:

            latest_row = district_data.iloc[-1]

            production = float(
                latest_row["Production_Tonnes"]
            )

            selected_year = str(
                latest_row["Year"]
            )

            production_source = (
                "Latest available historical data"
            )


    temperature, rainfall, humidity = get_weather_values(

        district_data,

        data.weather_condition
    )


    stress = calculate_stress(

        district_data,

        production,

        temperature,

        rainfall,

        humidity
    )


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


    prediction = shortfall_model.predict(

        np.array(
            [shortfall_features]
        )

    )[0]


    prediction = int(prediction)


    reverse_label_map = {

        0: "Low",

        1: "Medium",

        2: "High"
    }


    if (
        isinstance(metadata, dict)
        and "reverse_label_map" in metadata
    ):

        reverse_label_map = {

            int(k): v

            for k, v in metadata[
                "reverse_label_map"
            ].items()
        }


    shortfall_risk = reverse_label_map.get(
        prediction,
        "Unknown"
    )


    ai_recommendations = generate_ai_recommendations(

        production_stress=
            stress["production_stress"],

        temperature_stress=
            stress["temperature_stress"],

        rainfall_stress=
            stress["rainfall_stress"],

        humidity_stress=
            stress["humidity_stress"],

        weather_stress=
            stress["weather_stress"],

        equipment_risk=
            equipment_risk,

        shortfall_risk=
            shortfall_risk
    )


    ai_recommendation = generate_gemini_recommendation(

        state=data.state,

        district=data.district,

        production_risk=shortfall_risk,

        equipment_risk=
            round(equipment_risk, 2),

        weather_stress=
            round(stress["weather_stress"], 2),

        production_stress=
            round(stress["production_stress"], 2),

        trend=production_source
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


    return {

        "state": data.state,

        "district": data.district,


        "production_used": {

            "production_tonnes":
                round(production, 2),

            "year":
                selected_year,

            "source":
                production_source
        },


        "weather_condition":
            data.weather_condition,


        "weather_values": {

            "temperature_c":
                round(temperature, 2),

            "rainfall_mm":
                round(rainfall, 2),

            "humidity_percent":
                round(humidity, 2)
        },


        "stress_analysis":
            stress,


        "equipment_breakdown_risk_percent":
            round(equipment_risk, 2),


        "production_shortfall_risk":
            shortfall_risk,


        "ai_recommendations":
            ai_recommendations,


        "ai_recommendation":
            ai_recommendation,


        "recommendation_source":
            recommendation_source
    }