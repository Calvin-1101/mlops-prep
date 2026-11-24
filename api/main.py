"""
Serve the model via FastAPI 
"""

import os 
from contextlib import asynccontextmanager

import mlflow
import pandas as pd 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field 

import logging 


TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/california-housing@champion")
logger = logging.getLogger(__name__)
#Request/Response schemas 

class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median income of the block group, in $10k")
    HouseAge: float = Field(ge=0, le=100, description="Median house age, years")
    AveRooms: float = Field(gt=0, description="Average rooms per household")
    AveBedrms: float = Field(gt=0, description="Average bedrooms per household")
    Population: float = Field(ge=0, description="Block group population")
    AveOccup: float = Field(gt=0, description="Average household members")
    Latitude: float = Field(ge=32, le=42, description="California only")
    Longitude: float = Field(ge=-125, le=-114, description="California only")


class PredictRequest(BaseModel):
    features: list[HouseFeatures] = Field(min_length=1, max_length = 1000)


class PredictResponse(BaseModel):
    preds: list[float] 
    model_uri: str 


#model loading 
def load_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.pyfunc.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(app: FastAPI): 
    #runs once at startup -> loads the model a single time not on every request 
    try:
        app.state.model = load_model() 
    except Exception: 
        logger.exception("Failed to load model from %s", MODEL_URI)
        app.state.model = None 
    yield 

def model_loaded() -> bool:
    return getattr(app.state,'model',None) is not None 

app = FastAPI(title = "California Housing API", lifespan = lifespan)

#endpoints 
@app.get("/health")
def health() -> dict: 
    return {"status": "ok"}

@app.get("/ready")
def ready(): 
    if not model_loaded():
        raise HTTPException(status_code=503, detail ="model not loaded")

    return {"status": "ready", "model_uri": MODEL_URI}



@app.post("/predict", response_model = PredictResponse)
def predict(request:PredictRequest) -> PredictResponse:
    if not model_loaded():
        raise HTTPException(status_code=503, detail='model not loaded')
    df = pd.DataFrame([row.model_dump() for row in request.features])
    prediction = app.state.model.predict(df)
    return PredictResponse(preds=[float(p) for p in prediction], model_uri=MODEL_URI)