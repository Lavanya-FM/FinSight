import joblib
import os
import numpy as np
import pandas as pd

def load_models(path: str):
    models = {}
    for file in os.listdir(path):
        if file.endswith(".pkl"):
            model_name = file.replace(".pkl", "")
            models[model_name] = joblib.load(os.path.join(path, file))
    return models

def run_inference(model, data: dict):
    # Convert dict → DataFrame
    df = pd.DataFrame([data])
    # Run prediction
    pred = model.predict(df)
    # If probability available
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df).tolist()
        return {"prediction": pred.tolist(), "probability": prob}
    return {"prediction": pred.tolist()}
