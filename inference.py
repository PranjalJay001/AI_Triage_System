"""
inference.py — Hospital Triage OpenEnv Baseline Agent

Runs an LLM-based agent against all 3 triage tasks and emits structured logs.

MANDATORY ENV VARS:
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — model identifier
  HF_TOKEN      — Hugging Face / API key

STDOUT FORMAT (strictly followed):
  [START] task=<name> env=hospital_triage model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<bool> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
import textwrap
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf_placeholder")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK = "hospital_triage"
MAX_STEPS = 40
TEMPERATURE = 0.2
MAX_TOKENS = 400
SUCCESS_SCORE_THRESHOLD = 0.4

TASKS_TO_RUN = ["easy_triage", "medium_triage", "hard_triage"]


# ─────────────────────────────────────────────
#  Logging helpers (MANDATORY FORMAT)
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # sanitize action string — no newlines
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────
#  Environment HTTP Client
# ─────────────────────────────────────────────

class TriageEnvClient:
    def __init__(self, base_url: str, session_id: str = "default"):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id

    def reset(self, task_name: str, seed: int = 42) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task_name": task_name, "seed": seed, "session_id": self.session_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        payload = {**action, "session_id": self.session_id}
        resp = requests.post(f"{self.base_url}/step", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/state", params={"session_id": self.session_id}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def grade(self) -> float:
        resp = requests.get(f"{self.base_url}/grade", params={"session_id": self.session_id}, timeout=15)
        resp.raise_for_status()
        return resp.json()["score"]


# ─────────────────────────────────────────────
#  LLM Agent
# ─────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an experienced emergency department triage nurse AI agent.

Your job is to:
1. Assess patients and assign the correct triage category
2. Admit critical patients to the right resource type
3. Discharge stable patients to free up capacity
4. Minimize wait times for life-threatening cases

TRIAGE CATEGORIES:
- immediate: Life-threatening (SpO2 < 90%, GCS ≤ 8, BP < 90 systolic, etc.)
- urgent: Serious but stable (SpO2 91-93%, HR >120, fever >39°C)
- semi_urgent: Moderate (stable vitals, pain 3-6)
- non_urgent: Minor (normal vitals, pain ≤ 3)

RESOURCE TYPES:
- icu_bed: For IMMEDIATE category patients
- emergency_bed: For URGENT patients
- general_bed: For SEMI_URGENT and NON_URGENT patients
- ventilator: For patients with severe respiratory failure
- xray: Imaging
- lab: Blood tests

ACTION FORMAT — respond ONLY with a valid JSON object (no markdown, no explanation):
{
  "action_type": "assign_triage" | "admit_to_bed" | "discharge" | "order_test" | "escalate" | "wait",
  "patient_id": "P001",           (required for most actions)
  "triage_category": "immediate", (for assign_triage)
  "resource_id": "icu_bed_1",     (for admit_to_bed — use exact resource IDs from available_resources)
  "test_type": "ecg"              (for order_test)
}

STRATEGY:
- Always triage IMMEDIATE patients first
- Admit IMMEDIATE patients to icu_bed resources
- Admit URGENT patients to emergency_bed resources
- Discharge NON_URGENT patients if beds are needed
- Be efficient: process the most critical untriaged patient each step
""").strip()


def build_prompt(obs: Dict[str, Any], step: int) -> str:
    current = obs.get("current_patient")
    waiting = obs.get("waiting_patients", [])
    available = obs.get("available_resources", {})
    admitted = obs.get("admitted_patients", [])

    lines = [
        f"Step {step} | Time: {obs.get('current_time', 0):.0f} min | Queue: {obs.get('queue_length', 0)} patients",
        f"Critical patients waiting: {obs.get('critical_patients_waiting', 0)}",
        f"Avg wait time: {obs.get('avg_wait_time', 0):.1f} min",
        f"Last action result: {obs.get('last_action_result', 'N/A')}",
        "",
        "AVAILABLE RESOURCES:",
    ]
    for rtype, count in available.items():
        lines.append(f"  {rtype}: {count} free")

    if current:
        vitals = current.get("vital_signs", {})
        lines += [
            "",
            "CURRENT PATIENT TO TRIAGE:",
            f"  ID: {current['patient_id']} | Age: {current['age']}",
            f"  Complaint: {current['chief_complaint']}",
            f"  Heart Rate: {vitals.get('heart_rate', '?')} bpm | BP: {vitals.get('bp_systolic', '?')} mmHg",
            f"  SpO2: {vitals.get('spo2', '?')}% | Temp: {vitals.get('temperature', '?')}°C | RR: {vitals.get('resp_rate', '?')}/min",
            f"  Pain: {current.get('pain_score', '?')}/10",
            f"  Triaged: {current.get('assigned_triage', 'NOT YET')}",
        ]

    # Show triaged but not yet admitted patients
    triaged_not_admitted = [
        p for p in waiting
        if p.get("assigned_triage") and not p.get("is_admitted")
    ]
    if triaged_not_admitted:
        lines.append("")
        lines.append("TRIAGED BUT NOT YET ADMITTED:")
        for p in triaged_not_admitted[:5]:
            lines.append(f"  {p['patient_id']} | {p.get('assigned_triage')} | {p['chief_complaint'][:50]}")

    # Show untriaged patients
    untriaged = [p for p in waiting if not p.get("assigned_triage")]
    if untriaged:
        lines.append("")
        lines.append(f"UNTRIAGED WAITING ({len(untriaged)} patients):")
        for p in untriaged[:3]:
            vitals = p.get("vital_signs", {})
            lines.append(
                f"  {p['patient_id']} | Age {p['age']} | {p['chief_complaint'][:50]} | "
                f"HR:{vitals.get('heart_rate','?')} SpO2:{vitals.get('spo2','?')}%"
            )

    lines.append("")
    lines.append("What is your next action? Respond with JSON only.")
    return "\n".join(lines)


def get_agent_action(client: OpenAI, obs: Dict[str, Any], step: int, history: List[str]) -> Dict[str, Any]:
    user_prompt = build_prompt(obs, step)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        action = json.loads(raw)
        return action
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e} | raw={raw[:200]}", flush=True)
        return {"action_type": "wait"}
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return {"action_type": "wait"}


# ─────────────────────────────────────────────
#  Run one task episode
# ─────────────────────────────────────────────

def run_task(client: OpenAI, env_client: TriageEnvClient, task_name: str) -> Dict[str, Any]:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    try:
        reset_result = env_client.reset(task_name=task_name, seed=42)
        obs = reset_result["observation"]
        done = reset_result.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = get_agent_action(client, obs, step, history)
            action_str = json.dumps(action, separators=(",", ":"))

            try:
                step_result = env_client.step(action)
                obs = step_result["observation"]
                reward = float(step_result.get("reward", 0.0))
                done = step_result.get("done", False)
                error = obs.get("last_action_error")
            except Exception as e:
                reward = 0.0
                done = False
                error = str(e)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action_str[:80]} -> reward {reward:+.2f}")

            if done:
                break

        # Get final graded score
        score = env_client.grade()
        score = min(max(float(score), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_name} failed: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "score": score, "success": success, "steps": steps_taken}


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def wait_for_server(base_url: str, max_wait: int = 60) -> bool:
    print(f"[DEBUG] Waiting for server at {base_url} ...", flush=True)
    for _ in range(max_wait):
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                print("[DEBUG] Server is ready.", flush=True)
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    # Start the environment server in a subprocess if running locally
    import subprocess
    import sys

    server_proc = None
    if os.getenv("START_SERVER", "1") == "1":
        print("[DEBUG] Starting environment server...", flush=True)
        server_proc = subprocess.Popen(
            [sys.executable, "server.py"],
            env={**os.environ, "PORT": "7860"},
        )
        if not wait_for_server(ENV_BASE_URL, max_wait=30):
            print("[DEBUG] Server failed to start!", flush=True)
            if server_proc:
                server_proc.terminate()
            return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    results = []

    try:
        for task_name in TASKS_TO_RUN:
            session_id = f"session_{task_name}"
            env_client = TriageEnvClient(base_url=ENV_BASE_URL, session_id=session_id)
            result = run_task(client, env_client, task_name)
            results.append(result)
            print("", flush=True)  # blank line between tasks

        # Summary
        print("[DEBUG] ===== FINAL SUMMARY =====", flush=True)
        for r in results:
            status = "PASS" if r["success"] else "FAIL"
            print(f"[DEBUG] {r['task']:25s} | score={r['score']:.3f} | steps={r['steps']} | {status}", flush=True)
        avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
        print(f"[DEBUG] AVERAGE SCORE: {avg_score:.3f}", flush=True)

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()


if __name__ == "__main__":
    main()
