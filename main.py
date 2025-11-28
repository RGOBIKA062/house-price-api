from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import gradio as gr
from datetime import datetime
import logging

# Logging (initialize early so load errors get recorded)
logging.basicConfig(filename="app.log", level=logging.INFO)

# Try to load model but don't crash the app if loading fails (e.g., pickle/python mismatch)
model = None
try:
    model = joblib.load("house_model.pkl")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.exception("Failed to load model: %s", e)

# App initialization
app = FastAPI(title="House Price Predictor", version="1.1")

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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")

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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")

    preds = model.predict(batch.data)
    return {"predictions": preds.tolist()}

# Error Handler
@app.exception_handler(Exception)
def all_errors(request, exc):
    return {"error": str(exc)}

# Gradio Function
def predict_gradio(*inp):
    if model is None:
        return "Model not available"

    pred = model.predict([list(inp)])
    return float(pred[0])

# Gradio Interface
demo = gr.Interface(
    fn=predict_gradio,
    inputs=[gr.Number(label=f"Feature {i+1}") for i in range(8)],
    outputs=gr.Number(label="Predicted Price"),
    title="🏡 House Price Predictor"
)

# Mount Gradio into the FastAPI app so FastAPI serves Gradio's assets
# This avoids calling `demo.launch()` inside a request handler which can
# cause asset requests to return 404 when the Gradio server isn't available.
try:
    gr.mount_gradio_app(app, demo, path="/")
except Exception:
    # If mounting fails (older gradio versions), keep a simple root route
    # that informs the user the API is running.
    @app.get("/")
    def home():
        return {"status": "API running. Gradio UI unavailable."}