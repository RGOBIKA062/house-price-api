from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import gradio as gr
from datetime import datetime
import logging

# Load model
model = joblib.load("house_model.pkl")

# App initialization
app = FastAPI(title="House Price Predictor", version="1.1")

# Logging
logging.basicConfig(filename="app.log", level=logging.INFO)

# CORS (required for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Model
class Input(BaseModel):
    data: list[float]

# Batch Input
class BatchInput(BaseModel):
    data: list[list[float]]

# Health Check
@app.get("/health")
def health():
    return {"status": "OK"}

# Model Info Route
@app.get("/model-info")
def model_info():
    return {
        "model_name": "Linear Regression (California Housing)",
        "version": "1.1",
        "features_required": 8
    }

# Single Prediction
@app.post("/predict")
def predict(input: Input):
    pred = model.predict([input.data])
    output = float(pred[0])

    logging.info(f"Input: {input.data}, Output: {output}")

    return {
        "prediction": output,
        "timestamp": datetime.now().isoformat()
    }

# Batch Prediction
@app.post("/batch-predict")
def batch_predict(batch: BatchInput):
    preds = model.predict(batch.data)
    return {"predictions": preds.tolist()}

# Error Handler
@app.exception_handler(Exception)
def all_errors(request, exc):
    return {"error": str(exc)}

# Gradio Function
def predict_gradio(*inp):
    pred = model.predict([list(inp)])
    return float(pred[0])

# Gradio Interface
demo = gr.Interface(
    fn=predict_gradio,
    inputs=[gr.Number(label=f"Feature {i+1}") for i in range(8)],
    outputs=gr.Number(label="Predicted Price"),
    title="🏡 House Price Predictor"
)

# Route for Gradio UI
@app.get("/")
def home():
    return demo.launch(share=False, inline=True)