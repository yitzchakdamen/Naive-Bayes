from fastapi import FastAPI, HTTPException
from typing import List, Dict, Optional
import os
import pickle
from Model import ModelSystem
from pydantic import BaseModel
from app_models import ModelInfoResponse, PredictionRequest

app = FastAPI()

MODELS_DIR = "./Files_model"

model_system = ModelSystem()
model_system.upload_data(
    file="agaricus-lepiota.csv",
    target_variable="p",
    str_yes="e",
    str_no="p")
                
model_system.training("name_model")
model_system.upload_model(f"{MODELS_DIR}/name_model_training_75.json")
results:dict = model_system.testing()
print(results)

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
    uvicorn.run(app, host="0.0.0.0", port=8030)