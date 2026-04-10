from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from supermart_env import (
    SupermarketEnv,
    PRODUCT_CATALOGUE,
    ALL_PRODUCTS,
    ACTION_NAMES,
    N_ACTIONS,
    LEVEL_CONFIG,
)

app = FastAPI(
    title="SupermarketNav OpenEnv",
    description="Client-Server RL environment for LLM agent evaluation.",
    version="1.0.0",
)


_sessions: Dict[str, SupermarketEnv] = {}





class ResetRequest(BaseModel):
    level:    str            = Field("easy", description="easy | medium | hard")
    products: Optional[List[str]] = Field(
        None,
        description=(
            "Optional list of exact product names to use. "
            "Leave null to let the env pick randomly."
        ),
    )
    seed: Optional[int] = Field(None, description="RNG seed for reproducibility.")


class StepRequest(BaseModel):
    action: int = Field(
        ...,
        ge=0,
        le=N_ACTIONS - 1,
        description=f"Integer action 0-{N_ACTIONS - 1}. Names: {ACTION_NAMES}",
    )


class StateResponse(BaseModel):
    session_id:       str
    task_status:      str
    phase:            int
    phase_label:      str
    agent_pos:        List[int]
    targets:          List[str]
    inventory:        List[str]
    closest_target:   Optional[str]
    steps_taken:      int
    steps_remaining:  int
    total_reward:     float
    normalised_score: float
    action_mask:      List[bool]
    level:            str
    closest_rule:     bool
    event:            str


def _env_state(session_id: str, env: SupermarketEnv, event: str = "") -> Dict[str, Any]:
    info = env._build_info(event, env._total_reward)
    return {
        "session_id":       session_id,
        "task_status":      info["task_status"],
        "phase":            info["phase"],
        "phase_label":      info["phase_label"],
        "agent_pos":        info["agent_pos"],
        "targets":          info["targets"],
        "inventory":        info["inventory"],
        "closest_target":   info["closest_target"],
        "steps_taken":      info["steps_taken"],
        "steps_remaining":  info["steps_remaining"],
        "total_reward":     info["total_reward"],
        "normalised_score": info["normalised_score"],
        "action_mask":      info["action_mask"],
        "level":            info["level"],
        "closest_rule":     info["closest_rule"],
        "event":            event,
    }


def _get_env(session_id: str) -> SupermarketEnv:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return env




@app.get("/healthz", tags=["meta"])
def healthz():
    return {"status": "ok", "active_sessions": len(_sessions)}


@app.get("/catalogue", tags=["meta"])
def catalogue():

    return {"catalogue": PRODUCT_CATALOGUE, "all_products": list(ALL_PRODUCTS.keys())}


@app.post("/reset", tags=["env"])
def reset(body: Optional[ResetRequest] = None):
    # If the automated grader sends an empty request, use defaults
    if body is None:
        body = ResetRequest()
        
    if body.level not in LEVEL_CONFIG:
        raise HTTPException(
            status_code=422,
            detail=f"level must be one of {list(LEVEL_CONFIG)}",
        )

    session_id = str(uuid.uuid4())
    env = SupermarketEnv(level=body.level, products=body.products)

    try:
        _, info = env.reset(seed=body.seed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    _sessions[session_id] = env

    return {
        **_env_state(session_id, env, "Episode started"),
        "action_names": ACTION_NAMES,
        "level_config": {
            k: v for k, v in LEVEL_CONFIG[body.level].items()
            if k not in ("q_lr", "q_gamma", "eps_start", "eps_end", "eps_decay", "train_episodes")
        },
    }


@app.post("/step/{session_id}", tags=["env"])
def step(session_id: str, body: StepRequest):
    
    env = _get_env(session_id)

    if env.task_status != "In-Progress":
        raise HTTPException(
            status_code=409,
            detail=f"Episode already ended with status '{env.task_status}'. Call /reset to start a new one.",
        )

    obs, raw_reward, terminated, truncated, info = env.step(body.action)

    response = {
        **_env_state(session_id, env, info["event"]),
        "step_reward":  info["step_reward"],
        "terminated":   terminated,
        "truncated":    truncated,
        "done":         terminated or truncated,
    }


    if terminated or truncated:
        _sessions.pop(session_id, None)

    return response


@app.get("/state/{session_id}", tags=["env"])
def state(session_id: str):

    env = _get_env(session_id)
    return _env_state(session_id, env, "state query")
