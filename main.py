from fastapi import FastAPI, UploadFile, File
import joblib
import pandas as pd
import numpy as np
import io

app = FastAPI()

model = joblib.load("cart_abandonment_model (1).pkl")

@app.get("/")
def home():
    return {"message": "Cart Abandonment Predictor API is live"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    
    results = []
    for pred, prob in zip(predictions, probabilities):
        results.append({
            "prediction": int(pred),
            "result": "Likely to Convert" if pred == 1 else "Likely to Abandon",
            "conversion_probability": round(float(prob) * 100, 2)
        })
    
    return {"total_sessions": len(df), "predictions": results}