from fastapi import FastAPI, HTTPException
import requests
from Model import ModelSystem
from app_models import ModelInfoResponse, PredictionRequest

app = FastAPI()
model_system = ModelSystem()


@app.get("/api/models_info")
def get_models_info():
    try:
        return requests.get(model_system.GET_INFO_URL).json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prediction", response_model=dict)
def prediction(request: PredictionRequest):
    try:
        response = requests.get(f"{model_system.GET_FILE_URL}{request.model_name}.json")
        model_system.upload_model(response.json())
        return model_system.prediction(parameters=request.input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)