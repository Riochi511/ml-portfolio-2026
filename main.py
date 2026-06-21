from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def home():
    return {"message": "ML Portfolio API is live"}
@app.get("predict")
def predict():
    return {"prediction": "test response"}