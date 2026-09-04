from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AeroVoxel API",
    description="Software-defined aerodynamic simulation backend",
    version="0.1.0"
)

# Local-dev CORS. allow_origins=["*"] must not pair with allow_credentials=True
# (browsers reject that combo). This API does not use cookies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local educational prototype
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routes import demo_cases, upload, simulate, surrogate, recon

app.include_router(demo_cases.router)
app.include_router(upload.router)
app.include_router(simulate.router)
app.include_router(surrogate.router)
app.include_router(recon.router)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": "0.1.0",
        "message": "AeroVoxel backend is running smoothly"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
