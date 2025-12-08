import os
from fastapi import FastAPI
from src import inference
from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: int

app = FastAPI()
classifier, encoder = inference.load_models(
    classifier_path=os.path.join("models", "classifier.joblib"),
    encoder_path=os.path.join("models", "sentence_transformer.model"),
)

@app.get("/")
def welcome_root():
    return {"message": "Welcome to the ML API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    pred = inference.predict(classifier, encoder, request.text)
    return PredictResponse(prediction=pred)
