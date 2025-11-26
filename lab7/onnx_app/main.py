import time
import onnxruntime as ort

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-mpnet-base-cos-v1")
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session = ort.InferenceSession("model_optimized.onnx", sess_options=options, providers=["CPUExecutionProvider"])


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    embedding: list[float]
    server_time: float


@app.post("/predict")
def predict(req: PredictRequest) -> PredictResponse:
    t = time.perf_counter()

    inputs = tokenizer(req.text, padding=True, truncation=True, return_tensors="np")
    input_feed = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    _ = session.run(None, input_feed)

    t = time.perf_counter() - t

    return PredictResponse(embedding=[1, 2, 3], server_time=t)