# backend/app/main.py
import os

# Fix common Windows SSL certificate issues with httpx/ollama clients
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("SSL_CERT_DIR", None)

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .database import engine
from . import models
from .routers import applicant

# Create tables (development only — use Alembic in production)
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="ATS Voice Agent Backend",
    description="AI-powered resume screening, question generation, and voice interview simulation using Ollama",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "applicants", "description": "Resume upload, screening, questions & interview flow"}
    ]
)

# Enable CORS (critical for frontend to work from different origin/port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← In production: change to your actual frontend domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router (prefix is already in applicant.router)
app.include_router(applicant.router)


@app.get("/")
def root():
    return {
        "message": "ATS Voice Agent Backend is running!",
        "version": app.version,
        "docs": "/docs",
        "status": "healthy"
    }


# Global exception handler (catches all unhandled errors)
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"[GLOBAL ERROR] {str(exc)}")  # Log to terminal
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )