"""
Run with: uvicorn api:app --reload
Then open http://localhost:8000 in Chrome (Web Speech API support required)
for the voice UI, or POST directly to /ask.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from app.agents import VoxOrchestrator
from app import db

app = FastAPI(title="Vox — Voice AI Clinic Copilot")
orchestrator = VoxOrchestrator()
db.init_db()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    request_id: str
    answer: str
    agent: str
    variant: str
    tool_calls: list
    latency_seconds: float
    input_tokens: int
    output_tokens: int


class FeedbackRequest(BaseModel):
    request_id: str
    rating: int  # 1 = thumbs up, 0 = thumbs down


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = orchestrator.run(request.query)
    return AskResponse(
        request_id=result.request_id,
        answer=result.answer,
        agent=result.agent,
        variant=result.variant,
        tool_calls=result.tool_calls,
        latency_seconds=result.latency_seconds,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    ok = db.record_feedback(request.request_id, request.rating)
    return {"recorded": ok}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
