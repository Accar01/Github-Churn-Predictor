import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from model import load_saved_model


app = FastAPI(title="Churn Predictor API")

model_bundle = load_saved_model()
model = model_bundle["model"]
selected_features = model_bundle["selected_features"]


class UserFeatures(BaseModel):
    days_since_last_activity: float
    account_age_days: float
    language_count: float
    has_bio: int
    has_company: int
    has_location: int
    recent_event_count: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }


@app.get("/features")
def features():
    return {
        "selected_features": selected_features
    }


@app.post("/predict")
def predict_churn(user: UserFeatures):
    input_data = {
        "days_since_last_activity": user.days_since_last_activity,
        "account_age_days": user.account_age_days,
        "language_count": user.language_count,
        "has_bio": user.has_bio,
        "has_company": user.has_company,
        "has_location": user.has_location,
        "recent_event_count": user.recent_event_count,
    }

    features_df = pd.DataFrame([input_data], columns=selected_features)

    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0][1]

    return {
        "churned": bool(prediction),
        "churn_probability": round(float(probability), 3)
    }