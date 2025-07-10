from fastapi import FastAPI, HTTPException
from typing import List, Dict, Optional
import os
import pickle
from Model.Upload import UploadData
from Model.model_info import ModelInfo
from pydantic import BaseModel


app = FastAPI()

MODELS_DIR = "./Files_model"


class ModelInfoResponse(BaseModel):
    name: Optional[str]
    columns: List[Dict[str, List[str]]]
    target_variable: Optional[List[str]]


@app.get("/models/", response_model=List[ModelInfoResponse])
def list_models():
    try:
        models = [
            ModelInfo(UploadData.upload(os.path.join(MODELS_DIR, file))).get_model_info() 
            for file in os.listdir(MODELS_DIR) if file.endswith(".json")
            ]
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}")
def get_model_info(model_name: str):
    model_path = os.path.join(MODELS_DIR, model_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="המודל לא נמצא")

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # נניח שלמודל יש מאפיין בשם 'description' או '__dict__'
        if hasattr(model, "description"):
            return {"name": model_name, "description": model.description}
        else:
            return {"name": model_name, "info": str(model)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"שגיאה בקריאת המודל: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)