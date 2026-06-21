# ML Portfolio API

A FastAPI-powered REST API serving a Random Forest model 
for E-Commerce Cart Abandonment prediction.

## 🚀 Live Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check |
| POST | /predict | Upload CSV, get predictions |

## 📊 Model
- Algorithm: Random Forest Classifier
- ROC-AUC: 0.9784
- Dataset: 580,000 rows

## 🛠 Setup
```bash
pip install -r requirements.txt
uvicorn main:app --reload
📥 How to Use
Send a POST request to /predict with a CSV file.
Returns prediction (0/1) and conversion probability % for each session.
👤 Author
Bright Alfred Riochi | AI/ML Engineer