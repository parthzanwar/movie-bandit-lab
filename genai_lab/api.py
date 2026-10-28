"""FastAPI backend and static JavaScript frontend for the GenAI extension."""

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from genai_lab.assistant import answer_question, provider_status
from genai_lab.bandit_tools import (
    POLICY_NAMES,
    PROFILE_USERS,
    compare_policies,
    profile_summary,
    round_decision,
    top_movies,
)
from environment import candidate_movies
from genai_lab.retrieval import knowledge_index


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Movie Bandit GenAI Lab",
    description="Course-aligned RAG tutor and read-only movie-bandit tools.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatTurn(BaseModel):
    role: str
    content: str = Field(max_length=1500)


class AskRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1500)
    profile_label: str
    policy: str
    rounds: int = Field(ge=10, le=500)
    history: List[ChatTurn] = Field(default_factory=list, max_length=8)

    @field_validator("rounds")
    @classmethod
    def horizon_in_tens(cls, value):
        if value % 10:
            raise ValueError("Rounds must be in steps of 10.")
        return value


def _check_profile(profile_label):
    if profile_label not in PROFILE_USERS:
        raise HTTPException(status_code=422, detail="Unknown viewer profile.")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/api/status")
def status():
    return {
        **provider_status(),
        "policies": POLICY_NAMES,
        "profiles": list(PROFILE_USERS),
        "candidate_count": len(candidate_movies),
        "simulator_note": "Rewards and regret are computed by the existing synthetic simulator.",
    }


@app.get("/api/overview")
def overview(
    profile_label: str = Query(default="Horror buff"),
    policy: str = Query(default="LinUCB"),
    rounds: int = Query(default=120, ge=10, le=500),
):
    _check_profile(profile_label)
    if rounds % 10:
        raise HTTPException(status_code=422, detail="Rounds must be in steps of 10.")
    if policy not in POLICY_NAMES:
        raise HTTPException(status_code=422, detail="Unknown policy.")
    return {
        "profile": profile_summary(profile_label),
        "top_movies": top_movies(profile_label, limit=5),
        "policy_results": compare_policies(profile_label, rounds),
        "selected_decision": round_decision(profile_label, policy, rounds, rounds),
        "rounds": rounds,
        "policy": policy,
    }


@app.post("/api/ask")
async def ask(request: AskRequest):
    _check_profile(request.profile_label)
    if request.policy not in POLICY_NAMES:
        raise HTTPException(status_code=422, detail="Unknown policy.")
    history = [turn.model_dump() for turn in request.history]
    result = await run_in_threadpool(
        answer_question,
        request.query.strip(),
        request.profile_label,
        request.policy,
        request.rounds,
        history,
    )
    return result


@app.get("/api/health")
def health():
    return {"ok": True, "knowledge_chunks": len(knowledge_index.chunks)}
