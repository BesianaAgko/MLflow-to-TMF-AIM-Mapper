from fastapi import FastAPI, Request, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional


router = APIRouter(prefix="/tmf-api/AiM/v4")

# 🧠 Store posted models in memory
models = {}

@router.post("/aiModelSpecification")
async def receive_model(request: Request):
    data = await request.json()
    print("\n📥 Received TMF JSON:\n", data, "\n")

    required_fields = ["name", "modelDataSheet"]
    for field in required_fields:
        if field not in data or data[field] is None:
            return JSONResponse(
                status_code=400,
                content={"error": f"Missing required field: {field}"}
            )

    model_id = data.get("id")
    if not model_id:
        return JSONResponse(status_code=400, content={"error": "Missing model id"})

    models[model_id] = data

    return JSONResponse(
        status_code=201,
        content={"status": "received", "message": "Mock accepted JSON", "id": model_id}
    )


# ✅ GET all models
@router.get("/aiModelSpecification")
async def get_all_models():
    return list(models.values())

# ✅ GET model by ID with fields filter
@router.get("/aiModelSpecification/{model_id}")
async def get_model_by_id(model_id: str, fields: Optional[str] = None):
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    model = models[model_id]
    if fields:
        field_list = [f.strip() for f in fields.split(",")]
        filtered = {k: v for k, v in model.items() if k in field_list}
        return filtered
    return model



# 🌐 FastAPI setup
app = FastAPI(
    title="Mock TMF API",
    description="Test TMF POST",
)

app.include_router(router)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

