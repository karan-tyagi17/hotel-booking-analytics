"""
Hotel Booking Analytics - FastAPI Backend
------------------------------------------
REST API that exposes the AI agent as web endpoints.

Run with: py -m uvicorn src.api:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.agent import ask_agent, get_kpi

# Initialize FastAPI app
app = FastAPI(
    title="Hotel Booking Analytics API",
    description="AI-powered hotel booking data analyst",
    version="1.0.0"
)

# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─── REQUEST MODELS ──────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class KPIRequest(BaseModel):
    metric: str
    hotel_filter: str = "All"

# ─── ENDPOINTS ───────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Hotel Booking Analytics API is running!",
        "endpoints": ["/ask", "/kpi", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    result = ask_agent(request.question)
    return result

@app.post("/kpi")
def get_kpi_endpoint(request: KPIRequest):
    result = get_kpi(request.metric, request.hotel_filter)
    return result