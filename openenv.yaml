from __future__ import annotations
import subprocess
import time
import argparse
import os
import re
import sys
from typing import Any, Dict, Optional

import requests
from openai import OpenAI


DEFAULT_BASE_URL = "http://localhost:7860"
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "COLLECT"]
ACTION_NAME_TO_ID = {name: idx for idx, name in enumerate(ACTION_NAMES)}

SYSTEM_PROMPT = """You are an expert shopping agent navigating a grid supermarket.
GRID LAYOUT (8 rows x 7 cols):
  - Spawn:    row 0, col 3
  - Counter:  row 7, col 3  (EXIT -- go here after collecting all items)
  - Aisles:   A1(1,1) A2(3,1) A3(5,1) A4(1,5) A5(3,5) A6(5,5)
ACTIONS (reply with EXACTLY the action name -- nothing else):
  UP      move one row up    (row - 1)
  DOWN    move one row down  (row + 1)
  LEFT    move one col left  (col - 1)
  RIGHT   move one col right (col + 1)
  COLLECT pick up item at current aisle (only when standing on an aisle that has a target)
RULES:
  1. In Phase 0: navigate to each target aisle and COLLECT all required items.
  2. In Phase 1: return to the Counter at (7, 3) to complete the task.
  3. COLLECT is only valid when action_mask[4] is true (you are on a target aisle).
  4. On HARD level you MUST collect the CLOSEST item first.
  5. Every step costs reward; reach the counter as fast as possible.
Reply with ONE word -- the action name. No explanation, no punctuation."""


def build_user_message(state: Dict[str, Any]) -> str:
    mask_str = ", ".join(
        f"{name}={'OK' if ok else 'BLOCKED'}"
        for name, ok in zip(ACTION_NAMES, state["action_mask"])
    )
    lines = [
        f"Position : {state['agent_pos']}",
        f"Phase    : {state['phase']} ({state['phase_label']})",
        f"Targets  : {state['targets']}",
        f"Inventory: {state['inventory']}",
        f"Remaining: {[t for t in state['targets'] if t not in state['inventory']]}",
        f"Closest  : {state['closest_target']}",
        f"Steps    : {state['steps_taken']} / {state['steps_taken'] + state['steps_remaining']}",
        f"Total rwd: {state['total_reward']:.2f}",
        f"Masks    : {mask_str}",
        f"Last evt : {state.get('event', '')}",
    ]
    return "\n".join(lines)


def parse_action(text: str) -> Optional[int]:
    clean = text.strip().upper()
    if clean in ACTION_NAME_TO_ID:
        return ACTION_NAME_TO_ID[clean]
    for name, idx in ACTION_NAME_TO_ID.items():
        if re.search(rf"\b{name}\b", clean):
            return idx
    return None


def log_start(session_id: str, level: str, model: str):
    print(f"[START] session_id={session_id} level={level} model={model}", flush=True)


def log_step(step_num: int, action_name: str, state: Dict[str, Any]):
    print(
        f"[STEP] step={step_num} action={action_name} "
        f"step_reward={state.get('step_reward', 0.0):.4f} "
        f"score={state.get('normalised_score', 0.0):.4f}",
        flush=True,
    )


def log_end(session_id: str, state: Dict[str, Any]):
    print(
        f"[END] session_id={session_id} status={state.get('task_status', 'Unknown')} "
        f"steps={state.get('steps_taken', 0)} "
        f"final_score={state.get('normalised_score', 0.0):.4f}",
        flush=True,
    )


def ensure_server_running(base_url: str):
    """Start uvicorn server if not already running."""
    try:
        r = requests.get(f"{base_url}/healthz", timeout=3)
        if r.status_code == 200:
            return
    except requests.RequestException:
        pass

    print("[INFO] Starting server...", flush=True)
    subprocess.Popen(
        ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    for _ in range(20):
        time.sleep(1)
        try:
            r = requests.get(f"{base_url}/healthz", timeout=2)
            if r.status_code == 200:
                print("[INFO] Server is up.", flush=True)
                return
        except requests.RequestException:
            continue
    raise RuntimeError("Server failed to start within 20 seconds")


def run_agent(level: str, base_url: str, products=None, seed=None):
    ensure_server_running(base_url)

    api_base_url  = os.environ.get("API_BASE_URL", "").strip()
    model         = os.environ.get("MODEL_NAME", "gpt-4o-mini").strip()
    hf_token      = os.environ.get("HF_TOKEN", "").strip()
    effective_key = hf_token if hf_token else "placeholder-key"

    try:
        client = OpenAI(
            api_key=effective_key,
            base_url=api_base_url if api_base_url else None,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to create OpenAI client: {exc}", flush=True)
        sys.exit(1)

    reset_payload: Dict[str, Any] = {"level": level}
    if products:
        reset_payload["products"] = products
    if seed is not None:
        reset_payload["seed"] = seed

    try:
        r = requests.post(f"{base_url}/reset", json=reset_payload, timeout=10)
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"[ERROR] Could not reach server at {base_url}: {exc}", flush=True)
        sys.exit(1)

    state      = r.json()
    session_id = state["session_id"]
    step_num   = 0
    conversation: list = []

    log_start(session_id, level, model)

    while (
        not state.get("terminated", False)
        and not state.get("truncated", False)
        and not state.get("done", False)
        and state.get("task_status") == "In-Progress"
    ):
        user_msg = build_user_message(state)
        conversation.append({"role": "user", "content": user_msg})
        trimmed_history = conversation[-20:]

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + trimmed_history,
                max_tokens=10,
                temperature=0.0,
            )
            reply = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"[ERROR] LLM call failed: {exc}", flush=True)
            break

        action_id = parse_action(reply)
        if action_id is None:
            action_id = 1

        action_name = ACTION_NAMES[action_id]
        conversation.append({"role": "assistant", "content": action_name})

        try:
            r = requests.post(
                f"{base_url}/step/{session_id}",
                json={"action": action_id},
                timeout=10,
            )
            r.raise_for_status()
        except requests.RequestException as exc:
            print(f"[ERROR] Step request failed: {exc}", flush=True)
            break

        state    = r.json()
        step_num += 1
        log_step(step_num, action_name, state)

    log_end(session_id, state)

    # Clamp score strictly between 0 and 1 as required by the evaluator
    raw_score = state.get("normalised_score", 0.0)
    return max(0.001, min(0.999, raw_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM agent for SupermarketNav")
    parser.add_argument("--level", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, dest="base_url")
    parser.add_argument("--products", nargs="*", help="Optional product names")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    try:
        score = run_agent(
            level=args.level,
            base_url=args.base_url,
            products=args.products,
            seed=args.seed,
        )
        print(f"[DONE] score={score:.4f}", flush=True)
    except Exception as exc:
        print(f"[FATAL] {exc}", flush=True)
        sys.exit(1)

    sys.exit(0)
