import joblib
from fastapi import FastAPI, Request
import uvicorn
import pandas as pd
from typing import Dict, Any

app = FastAPI(title="Obesity Prediction API")

# =============================
# Load trained model
# =============================
MODEL_PATH = "logreg_obesity_multiclass.joblib"
model = joblib.load(MODEL_PATH)

# =============================
# Features used during training
# =============================
FEATURES = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC',
    'FAF', 'TUE', 'CALC', 'MTRANS'
]

CLASS_LABELS = {
    0: "Normal_Weight",
    1: "Overweight_Level_I",
    2: "Overweight_Level_II",
    3: "Obesity_Type_I",
    4: "Insufficient_Weight",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III"
}


def predict_single(patient: Dict[str, Any]):
    # Convert dict to DataFrame
    X = pd.DataFrame([patient])

    # Check for missing features
    missing = [f for f in FEATURES if f not in X.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Keep only features in the correct order
    X = X[FEATURES]

    # Convert all values to float
    X = X.astype(float)

    # Make predictions
    pred_class = int(model.predict(X)[0])
    pred_proba = model.predict_proba(X)[0].tolist()
    pred_label = CLASS_LABELS[pred_class]

    # Top 3 predictions
    top3_idx = sorted(range(len(pred_proba)), key=lambda i: pred_proba[i], reverse=True)[:3]
    top3 = [{"label": CLASS_LABELS[i], "probability": pred_proba[i]} for i in top3_idx]

    return {
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "probabilities": pred_proba,
        "top3": top3
    }


@app.post("/predict")
async def predict(patient: Dict[str, Any]):
    try:
        return predict_single(patient)
    except Exception as e:
        # Return error instead of crashing
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
