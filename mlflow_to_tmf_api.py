from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlflow_to_tmf import dynamic_mlflow_to_tmf

app = FastAPI()

class MapRequest(BaseModel):
    run_id: str
    tracking_uri: str = "http://127.0.0.1:5000"

class MapResponse(BaseModel):
    status: str
    data: dict

@app.post("/map", response_model=MapResponse)
async def map_mlflow_to_tmf(request: MapRequest):
    try:
        tmf_json = dynamic_mlflow_to_tmf(request.run_id, request.tracking_uri)
        return MapResponse(status="success", data=tmf_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Mapping failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "tmf-api"}