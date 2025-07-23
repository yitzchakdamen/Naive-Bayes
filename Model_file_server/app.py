from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from Model import ModelSystem

app = FastAPI()

UPLOAD_FOLDER = Path("models")
UPLOAD_FOLDER.mkdir(exist_ok=True)
model_system = ModelSystem()



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = UPLOAD_FOLDER / file.filename
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    return {"message": f"הקובץ '{file.filename}' נשמר בהצלחה"}


@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = UPLOAD_FOLDER / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="הקובץ לא נמצא")
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')

@app.get("/models_info")
def get_models_info():
    try:
        return model_system.get_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)