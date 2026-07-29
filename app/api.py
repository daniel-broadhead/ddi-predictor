from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict_from_names

app = FastAPI()


class DrugPair(BaseModel):
    drug1: str
    drug2: str


@app.post('/predict')
def predict(pair: DrugPair):
    try:
        label, top_features = predict_from_names(pair.drug1, pair.drug2)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        'interaction_type': label,
        'top_features': top_features
    }