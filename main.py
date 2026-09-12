from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import requests
from pathlib import Path
import os
import ee
from dotenv import load_dotenv
from google import genai
import sqlite3
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sentinel2 import (
    initialize_earth_engine,
    generate_manganese_map
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
# =========================================================
# AUTHENTICATION
# =========================================================

DB_PATH = BASE_DIR / "users.db"

SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "change-this-secret-key-in-production"
)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/login"
)
def init_database():
    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_database()
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
# INITIALIZE GOOGLE EARTH ENGINE
# =========================================================

EARTH_ENGINE_AVAILABLE = initialize_earth_engine()

print(
    "Earth Engine available:",
    EARTH_ENGINE_AVAILABLE
)
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
    equipment_mode: str = "Auto"
    production_tonnes: float | None = None
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str
def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str):
    return pwd_context.verify(password, hashed_password)


def create_access_token(data: dict):
    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    to_encode.update({
        "exp": expire
    })

    return jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )
# =========================================================
# SENTINEL MAP REQUEST MODEL
# =========================================================

class MapRequest(BaseModel):

    west: float

    south: float

    east: float

    north: float
# =========================================================
# LIVE WEATHER FROM OPEN-METEO
# =========================================================

def get_live_weather(state, district):

    try:

        # -------------------------------------------------
        # STEP 1: Convert district name to coordinates
        # -------------------------------------------------

        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"

        geocode_params = {
            "name": f"{district}, {state}",
            "count": 10,
            "language": "en",
            "format": "json",
            "countryCode": "IN"
        }

        geocode_response = requests.get(
            geocode_url,
            params=geocode_params,
            timeout=4
        )

        geocode_response.raise_for_status()

        geocode_data = geocode_response.json()

        results = geocode_data.get("results", [])

        if not results:
            raise Exception(
                f"Location not found: {district}, {state}"
            )

        location = results[0]

        latitude = float(location["latitude"])
        longitude = float(location["longitude"])

        # -------------------------------------------------
        # STEP 2: Get live weather
        # -------------------------------------------------

        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "precipitation"
            ),
            "timezone": "auto"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=4
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        current = weather_data.get("current", {})

        temperature = float(
            current.get("temperature_2m", 0)
        )

        humidity = float(
            current.get("relative_humidity_2m", 0)
        )

        rainfall = float(
            current.get("precipitation", 0)
        )

        print(
            f"✓ Live weather: "
            f"{district}, {state} | "
            f"Temperature={temperature}°C | "
            f"Rainfall={rainfall} mm | "
            f"Humidity={humidity}%"
        )

        return (
            temperature,
            rainfall,
            humidity
        )

    except Exception as e:

        print("✗ LIVE WEATHER ERROR:")
        print(e)

        raise HTTPException(
            status_code=500,
            detail=f"Unable to fetch live weather: {str(e)}"
        )
# =========================================================
# WEATHER SCENARIO FUNCTION
# =========================================================

def get_weather_values(
    district_data,
    weather_condition,
    state=None,
    district=None
):

    weather_condition = weather_condition.lower().strip()
    if weather_condition == "auto":

        return get_live_weather(
            state,
            district
        )

    avg_temperature = float(
        district_data["Avg_Temperature_C"].mean()
    )

    avg_rainfall = float(
        district_data["Total_Rainfall_mm"].mean()
    )

    avg_humidity = float(
        district_data["Avg_Humidity_pct"].mean()
    )
    if weather_condition == "auto":

        try:
            return get_live_weather(
                state,
                district
            )

        except Exception:
            print(
                "⚠ Live weather unavailable."
                " Using offline district data."
            )

            return (
                avg_temperature,
                avg_rainfall,
                avg_humidity
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

# =========================================================
# FIXED: WEATHER SCENARIO FUNCTION
# =========================================================

def get_weather_values(
    district_data,
    weather_condition,
    state=None,
    district=None
):
    weather_condition = weather_condition.lower().strip()

    avg_temperature = float(district_data["Avg_Temperature_C"].mean())
    avg_rainfall = float(district_data["Total_Rainfall_mm"].mean())
    avg_humidity = float(district_data["Avg_Humidity_pct"].mean())

    if weather_condition == "auto":
        try:
            return get_live_weather(state, district)
        except Exception:
            print("⚠ Live weather unavailable. Using offline district data.")
            return avg_temperature, avg_rainfall, avg_humidity

    if weather_condition == "good":
        return (
            float(district_data["Avg_Temperature_C"].quantile(0.25)),
            float(district_data["Total_Rainfall_mm"].quantile(0.25)),
            float(district_data["Avg_Humidity_pct"].quantile(0.25))
        )
    elif weather_condition == "bad":
        return (
            float(district_data["Avg_Temperature_C"].quantile(0.75)),
            float(district_data["Total_Rainfall_mm"].quantile(0.75)),
            float(district_data["Avg_Humidity_pct"].quantile(0.75))
        )
    elif weather_condition == "worst":
        return (
            float(district_data["Avg_Temperature_C"].max()),
            float(district_data["Total_Rainfall_mm"].max()),
            float(district_data["Avg_Humidity_pct"].max())
        )

    return avg_temperature, avg_rainfall, avg_humidity


# =========================================================
# DYNAMIC MULTI-TIER LOCAL AI RECOMMENDATION ENGINE
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

    # Map all stress parameters with numerical values
    factors = {
        "Production Performance": production_stress,
        "Temperature Conditions": temperature_stress,
        "Rainfall Conditions": rainfall_stress,
        "Humidity Conditions": humidity_stress,
        "Overall Weather Stress": weather_stress,
        "Equipment Reliability": equipment_risk / 100.0
    }

    # Sort factors by severity
    sorted_factors = sorted(
        factors.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for factor, value in sorted_factors[:3]:
        risk_factors.append({
            "factor": factor,
            "severity_score": round(float(value), 3)
        })

    # =========================================================
    # 1. PRODUCTION STRESS (5 GRANULAR TIERS)
    # =========================================================
    prod_pct = int(production_stress * 100)

    if production_stress >= 0.75:
        recommendations.append({
            "priority": "Critical",
            "title": "Severe Production Deficit Emergency",
            "issue_detected": "Critical Production Drop",
            "why_it_matters": f"Production stress reached a critical level of {prod_pct}%. Tonnage is drastically below target baseline.",
            "recommended_actions": [
                "Halt non-essential operations and perform an immediate throughput audit.",
                "Deploy emergency haul fleet capacity to resolve primary pit bottlenecks.",
                "Re-evaluate daily target allocations with plant managers immediately."
            ]
        })
    elif production_stress >= 0.50:
        recommendations.append({
            "priority": "High",
            "title": "Substantial Yield Reduction",
            "issue_detected": "High Production Stress",
            "why_it_matters": f"Production stress is at {prod_pct}%, indicating significant lagging in pit-to-surface transport.",
            "recommended_actions": [
                "Audit primary crusher throughput to identify feed rate delays.",
                "Optimize haul truck dispatch cycles and minimize idle queuing times.",
                "Adjust shift handovers to eliminate operational downtime gaps."
            ]
        })
    elif production_stress >= 0.30:
        recommendations.append({
            "priority": "Medium-High",
            "title": "Moderate Production Shortfall",
            "issue_detected": "Moderate Output Variance",
            "why_it_matters": f"Production stress measured at {prod_pct}%. Tonnage is falling behind daily schedule quotas.",
            "recommended_actions": [
                "Increase monitoring of hourly excavator loading targets.",
                "Streamline internal pit traffic to prevent transport delays."
            ]
        })
    elif production_stress >= 0.15:
        recommendations.append({
            "priority": "Medium",
            "title": "Minor Tonnage Variance",
            "issue_detected": "Slight Production Deficit",
            "why_it_matters": f"Production stress is slightly elevated at {prod_pct}%. Small operational friction detected.",
            "recommended_actions": [
                "Track shift-wise output targets against weekly averages.",
                "Ensure maximum machine availability during peak operation hours."
            ]
        })

    # =========================================================
    # 2. HUMIDITY STRESS (4 TIERS)
    # =========================================================
    hum_pct = int(humidity_stress * 100)

    if humidity_stress >= 0.80:
        recommendations.append({
            "priority": "Critical",
            "title": "Extreme Humidity & Saturation Protocol",
            "issue_detected": "Severe Moisture Hazard",
            "why_it_matters": f"Humidity stress reached {hum_pct}%. Severe moisture risk for high-voltage systems and screening units.",
            "recommended_actions": [
                "Enforce IP65 electrical enclosure seals and deploy industrial desiccants.",
                "Run anti-clogging routines on damp ore screening decks immediately.",
                "Inspect motor insulation resistance across all dewatering pumps."
            ]
        })
    elif humidity_stress >= 0.55:
        recommendations.append({
            "priority": "High",
            "title": "High Moisture & Material Clogging Risk",
            "issue_detected": "Elevated Humidity Level",
            "why_it_matters": f"Humidity stress measured at {hum_pct}%, creating potential wet ore blinding on conveyor belts.",
            "recommended_actions": [
                "Inspect conveyor transfer chutes regularly for sticky manganese buildup.",
                "Check moisture traps on pneumatic lines and air compressors."
            ]
        })
    elif humidity_stress >= 0.35:
        recommendations.append({
            "priority": "Medium",
            "title": "Moderate Humidity Dampness",
            "issue_detected": "Moderate Environmental Moisture",
            "why_it_matters": f"Humidity stress is at {hum_pct}%. Minor moisture impact expected on outdoor machinery.",
            "recommended_actions": [
                "Conduct routine checks on exposed electrical control boxes.",
                "Apply anti-corrosive lubricants to exposed moving components."
            ]
        })

    # =========================================================
    # 3. RAINFALL STRESS (4 TIERS)
    # =========================================================
    rain_pct = int(rainfall_stress * 100)

    if rainfall_stress >= 0.75:
        recommendations.append({
            "priority": "Critical",
            "title": "Flash Inundation & Slope Hazard",
            "issue_detected": "Extreme Rainfall Impact",
            "why_it_matters": f"Rainfall stress index is at {rain_pct}%. High risk of pit floor flooding and bench instability.",
            "recommended_actions": [
                "Activate main stage high-volume pit dewatering pumps.",
                "Restrict heavy haulage along unpaved pit ramps and slippery inclines.",
                "Monitor slope stability sensors along high pit walls."
            ]
        })
    elif rainfall_stress >= 0.45:
        recommendations.append({
            "priority": "High",
            "title": "Haul Road Degradation Risk",
            "issue_detected": "Substantial Rainfall",
            "why_it_matters": f"Rainfall stress measured at {rain_pct}%, causing erosion along haul routes.",
            "recommended_actions": [
                "Apply coarse gravel to soft patches on primary haul roads.",
                "Clear roadside drainage ditches to clear runoff water rapidly."
            ]
        })
    elif rainfall_stress >= 0.20:
        recommendations.append({
            "priority": "Medium",
            "title": "Light Precipitation Impact",
            "issue_detected": "Low to Moderate Rain",
            "why_it_matters": f"Rainfall stress index is {rain_pct}%. Surface slipperiness may slightly slow transport.",
            "recommended_actions": [
                "Enforce strict speed limits for loaded haul trucks."
            ]
        })

    # =========================================================
    # 4. TEMPERATURE STRESS (4 TIERS)
    # =========================================================
    temp_pct = int(temperature_stress * 100)

    if temperature_stress >= 0.75:
        recommendations.append({
            "priority": "High",
            "title": "Extreme Thermal Overheating Risk",
            "issue_detected": "Severe Temperature Stress",
            "why_it_matters": f"Temperature stress reached {temp_pct}%. Heavy engine overheating risk for excavators.",
            "recommended_actions": [
                "Inspect radiator airflow and coolant levels during shift changes.",
                "Schedule high-demand earthmoving during morning hours."
            ]
        })
    elif temperature_stress >= 0.45:
        recommendations.append({
            "priority": "Medium",
            "title": "Elevated Engine Temperature",
            "issue_detected": "Moderate Heat Stress",
            "why_it_matters": f"Temperature stress measured at {temp_pct}%. Engine oil breakdown rates increase.",
            "recommended_actions": [
                "Monitor hydraulic fluid temperature gauges continuously."
            ]
        })

    # =========================================================
    # 5. EQUIPMENT RISK (4 TIERS)
    # =========================================================
    if equipment_risk >= 60.0:
        recommendations.append({
            "priority": "Critical",
            "title": "Imminent Breakdown Warning",
            "issue_detected": "Critical Equipment Failure Risk",
            "why_it_matters": f"Predicted breakdown risk is at {round(equipment_risk, 1)}%. High risk of unplanned line stoppage.",
            "recommended_actions": [
                "Pull high-risk machinery for emergency diagnostic inspection.",
                "Prepare standby units for rapid field replacement."
            ]
        })
    elif equipment_risk >= 30.0:
        recommendations.append({
            "priority": "High",
            "title": "High Wear & Mechanical Fatigue",
            "issue_detected": "Elevated Equipment Risk",
            "why_it_matters": f"Equipment breakdown risk measured at {round(equipment_risk, 1)}%.",
            "recommended_actions": [
                "Perform preventive lubrication and hydraulic filter replacement."
            ]
        })
    elif equipment_risk >= 15.0:
        recommendations.append({
            "priority": "Medium",
            "title": "Routine Wear Monitoring",
            "issue_detected": "Moderate Mechanical Stress",
            "why_it_matters": f"Equipment breakdown risk is at {round(equipment_risk, 1)}%.",
            "recommended_actions": [
                "Perform scheduled start-of-shift mechanical inspections."
            ]
        })

    # Fallback if no specific thresholds triggered
    if not recommendations:
        recommendations.append({
            "priority": "Low",
            "title": "Stable Baseline Operations",
            "issue_detected": "Normal Operational Parameters",
            "why_it_matters": "All measured stress indices are operating within safe baseline limits.",
            "recommended_actions": [
                "Maintain standard daily operational routines and safety monitoring."
            ]
        })

    top_factor = risk_factors[0]["factor"] if risk_factors else "Operational Parameters"

    if shortfall_risk == "High":
        ai_summary = f"CRITICAL: System predicts a HIGH production shortfall risk. Primary driver is {top_factor}."
    elif shortfall_risk == "Medium":
        ai_summary = f"WARNING: System predicts a MODERATE production shortfall risk. Primary area of focus is {top_factor}."
    else:
        ai_summary = f"STABLE: Overall production shortfall risk is LOW. Continue monitoring {top_factor}."

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
# SERVE MANGANESE MAP
# =========================================================

@app.get("/map")

def get_map():

    map_file = BASE_DIR / "moil_manganese_map.html"

    if not map_file.exists():

        raise HTTPException(
            status_code=404,
            detail="Map file not found"
        )

    return FileResponse(
        map_file,
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
# GENERATE SENTINEL-2 MANGANESE MAP
# =========================================================

@app.post("/generate-map")

def generate_map(
    data: MapRequest,
    current_user: dict = Depends(get_current_user)
):

    if not EARTH_ENGINE_AVAILABLE:

        raise HTTPException(

            status_code=503,

            detail="Google Earth Engine is not available"

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

            detail="Map has not been generated yet"

        )


    return FileResponse(

        map_path,

        media_type="text/html"

    )
# =========================================================
# MAIN PREDICTION
# =========================================================
@app.post("/register")
def register_user(data: RegisterRequest):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id FROM users WHERE email = ?",
        (data.email.lower().strip(),)
    )

    existing_user = cursor.fetchone()

    if existing_user:
        conn.close()

        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    hashed_password = hash_password(data.password)

    cursor.execute(
        """
        INSERT INTO users (name, email, password)
        VALUES (?, ?, ?)
        """,
        (
            data.name.strip(),
            data.email.lower().strip(),
            hashed_password
        )
    )

    conn.commit()
    conn.close()

    return {
        "status": "success",
        "message": "Registration successful"
    }
@app.post("/login")
def login_user(data: LoginRequest):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, name, email, password
        FROM users
        WHERE email = ?
        """,
        (data.email.lower().strip(),)
    )

    user = cursor.fetchone()

    conn.close()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    user_id, name, email, hashed_password = user

    if not verify_password(
        data.password,
        hashed_password
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    token = create_access_token({
        "sub": str(user_id),
        "email": email,
        "name": name
    })

    return {
        "status": "success",
        "access_token": token,
        "token_type": "bearer",
        "name": name
    }
def get_current_user(
    token: str = Depends(oauth2_scheme)
):

    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid or expired authentication token"
    )

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        user_id = payload.get("sub")

        if user_id is None:
            raise credentials_exception

        return payload

    except JWTError:

        raise credentials_exception
@app.post("/predict")
def predict(
    data: PredictionInput,
    current_user: dict = Depends(get_current_user)

):


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

        data.weather_condition,
        data.state,
        data.district
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


    # ==========================================
    # EQUIPMENT RISK
    # ==========================================

    if data.equipment_mode == "Auto":

        # Use trained ML model
        equipment_risk = float(
            equipment_model.predict(
                equipment_features
            )[0]
        )

        equipment_risk_source = "ML Prediction"


    elif data.equipment_mode == "Low":

        # Low equipment failure simulation
        equipment_risk = 10.0

        equipment_risk_source = "Simulation - Low"


    elif data.equipment_mode == "Medium":

        # Medium equipment failure simulation
        equipment_risk = 50.0

        equipment_risk_source = "Simulation - Medium"


    elif data.equipment_mode == "High":

        # High equipment failure simulation
        equipment_risk = 85.0

        equipment_risk_source = "Simulation - High"


    else:

        raise HTTPException(
            status_code=400,
            detail="Invalid equipment mode"
        )


    # Keep risk between 0 and 100
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
