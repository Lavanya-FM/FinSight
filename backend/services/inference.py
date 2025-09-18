import joblib
from pathlib import Path
import pandas as pd

# Define the models directory
models_dir = Path("backend/models")
models_dir.mkdir(exist_ok=True)

# Initialize models (use your trained models or parameters here)
models ={}

# Save models using joblib
for name, model in models.items():
    joblib.dump(model, models_dir / f"{name}.pkl")
    print(f"Saved {name}.pkl")

def run_inference(model, data: dict):
    try:
        df = pd.DataFrame([data])
        pred = model.predict(df)
        result = {"prediction": pred.tolist()}
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df).tolist()
            result["probability"] = prob
        return result
    except Exception as e:
        raise ValueError(f"Invalid input for model inference: {str(e)}")
