"""
Hospital Triage OpenEnv — FastAPI Server
Exposes /reset, /step, /state, /tasks, /grade endpoints.
"""

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import (
    ActionType,
    HospitalTriageEnv,
    TASKS,
    TriageAction,
    TriageCategory,
)

app = FastAPI(
    title="Hospital Triage OpenEnv",
    description="RL environment simulating emergency department patient triage.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One env instance per session (simple single-user approach)
_envs: Dict[str, HospitalTriageEnv] = {}


def get_env(session_id: str = "default") -> HospitalTriageEnv:
    if session_id not in _envs:
        _envs[session_id] = HospitalTriageEnv(task_name="easy_triage")
    return _envs[session_id]


# ─────────────────────────────────────────────
#  Request / Response Models
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: Optional[str] = "easy_triage"
    seed: Optional[int] = 42
    session_id: Optional[str] = "default"


class StepRequest(BaseModel):
    action_type: str
    patient_id: Optional[str] = None
    triage_category: Optional[str] = None
    resource_id: Optional[str] = None
    test_type: Optional[str] = None
    notes: Optional[str] = None
    session_id: Optional[str] = "default"


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Hospital Triage OpenEnv",
        "version": "1.0.0",
        "tasks": list(TASKS.keys()),
        "endpoints": ["/reset", "/step", "/state", "/grade", "/tasks", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    return {
        name: {
            "description": cfg["description"],
            "difficulty": cfg["difficulty"],
            "n_patients": cfg["n_patients"],
            "max_steps": cfg["max_steps"],
        }
        for name, cfg in TASKS.items()
    }


@app.post("/reset")
def reset(req: ResetRequest = None):
    if req is None:
        req = ResetRequest()
    task_name = req.task_name or "easy_triage"
    if task_name not in TASKS:
        raise HTTPException(400, f"Unknown task: {task_name}. Valid tasks: {list(TASKS.keys())}")
    env = HospitalTriageEnv(task_name=task_name, seed=req.seed or 42)
    _envs[req.session_id or "default"] = env
    result = env.reset()
    return result.model_dump()


@app.post("/step")
def step(req: StepRequest):
    env = get_env(req.session_id or "default")

    # Validate action_type
    try:
        action_type = ActionType(req.action_type)
    except ValueError:
        raise HTTPException(400, f"Invalid action_type: {req.action_type}. Valid: {[a.value for a in ActionType]}")

    # Validate triage_category if provided
    triage_cat = None
    if req.triage_category:
        try:
            triage_cat = TriageCategory(req.triage_category)
        except ValueError:
            raise HTTPException(400, f"Invalid triage_category: {req.triage_category}. Valid: {[t.value for t in TriageCategory]}")

    action = TriageAction(
        action_type=action_type,
        patient_id=req.patient_id,
        triage_category=triage_cat,
        resource_id=req.resource_id,
        test_type=req.test_type,
        notes=req.notes,
    )

    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    return result.model_dump()


@app.get("/state")
def state(session_id: str = Query(default="default")):
    env = get_env(session_id)
    return env.state()


@app.get("/grade")
def grade(session_id: str = Query(default="default")):
    env = get_env(session_id)
    score = env.grade()
    return {"score": score, "task": env.task_name}


@app.post("/grade")
def grade_post(body: Dict[str, Any] = None):
    session_id = (body or {}).get("session_id", "default")
    env = get_env(session_id)
    score = env.grade()
    return {"score": score, "task": env.task_name}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
