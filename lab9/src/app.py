import os.path

from src import inference
from fastapi import FastAPI

from src.api.models.iris import PredictRequest, PredictResponse

app = FastAPI()
model = inference.load_model(os.path.join("models", "iris.joblib"))


@app.get("/")
def welcome_root():
    return {"message": "Welcome to the ML API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    prediction = inference.predict(model, request.model_dump())
    return PredictResponse(prediction=prediction)
