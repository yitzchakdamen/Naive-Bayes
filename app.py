from fastapi import FastAPI, HTTPException
from typing import List, Dict, Optional
import os
import pickle
from Model.Upload import UploadData
from Model.model_info import ModelInfo
from Model.Model_System import ModelSystem
from pydantic import BaseModel
from Server.app_models import ModelInfoResponse, PredictionRequest

app = FastAPI()

MODELS_DIR = "./Files_model"
model_system = ModelSystem()


@app.get("/api/models_info/", response_model=List[ModelInfoResponse]) 
def models_info():
    try:
        return model_system.get_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prediction", response_model=dict)
def prediction(request: PredictionRequest):
    try:
        model_system.upload_model(os.path.join(MODELS_DIR, f"{request.model_name}.json"))
        return model_system.prediction(parameters=request.input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)