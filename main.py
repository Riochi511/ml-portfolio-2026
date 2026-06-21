from fastapi import FastAPI, UploadFile, File, HTTPException
import joblib
import pandas as pd
import numpy as np
import io

app = FastAPI()

model = joblib.load("cart_abandonment_model (1).pkl")

@app.get("/")
def home():
    return {"message": "ML Portfolio API is live"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, 
            detail="Only CSV files are accepted"
        )
    
    contents = await file.read()
    
    if len(contents) == 0:
        raise HTTPException(
            status_code=400, 
            detail="Uploaded file is empty"
        )
    
    df = pd.read_csv(io.BytesIO(contents))
    
    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="CSV file contains no data"
        )
    
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