# 🏥 Hospital Triage OpenEnv

> A real-world OpenEnv reinforcement learning environment simulating emergency department patient triage.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-brightgreen)](https://openenv.dev)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](./Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

In crowded emergency departments worldwide, triage nurses make life-or-death decisions under pressure: which patient needs immediate care? Where should the last ICU bed go? This environment simulates exactly that.

An AI agent acts as an emergency department triage officer. It must:
- **Assess** incoming patients using vital signs and chief complaints
- **Assign** correct triage categories (IMMEDIATE → NON_URGENT)
- **Allocate** scarce hospital resources (ICU beds, ventilators, emergency beds)
- **Discharge** stable patients to free capacity for critical cases
- **Minimize** wait times for life-threatening presentations

Unlike binary success/fail reward systems, every action yields **partial progress signals** — rewarding correct triage immediately, penalizing critical patients waiting too long, and incentivizing efficient resource matching.

---

## 🗂 Project Structure

```
hospital-triage-env/
├── environment.py      # Core OpenEnv environment (models, logic, grader)
├── server.py           # FastAPI server exposing /reset, /step, /state, /grade
├── inference.py        # Baseline LLM agent inference script
├── openenv.yaml        # OpenEnv spec metadata
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition for HF Spaces
└── README.md
```

---

## 🔌 Action & Observation Spaces

### Observation Space

| Field | Type | Description |
|---|---|---|
| `current_time` | float | Episode time in minutes |
| `waiting_patients` | list[Patient] | Patients in the waiting queue |
| `admitted_patients` | list[Patient] | Currently admitted patients |
| `available_resources` | dict | `{resource_type: count_free}` |
| `current_patient` | Patient \| null | Next patient pending triage |
| `last_action_result` | str | Human-readable result of last action |
| `last_action_error` | str \| null | Error message if action failed |
| `queue_length` | int | Number of waiting patients |
| `avg_wait_time` | float | Average wait across all waiting patients |
| `critical_patients_waiting` | int | IMMEDIATE severity patients not yet admitted |
| `step_number` | int | Current step index |

### Patient Object

```json
{
  "patient_id": "P001",
  "age": 67,
  "chief_complaint": "chest pain with ST elevation",
  "vital_signs": {
    "heart_rate": 145.2,
    "bp_systolic": 78.3,
    "spo2": 84.1,
    "temperature": 36.8,
    "resp_rate": 28.0
  },
  "pain_score": 9,
  "assigned_triage": null,
  "assigned_bed": null,
  "tests_ordered": [],
  "wait_time": 3.0
}
```

### Action Space

| `action_type` | Required Fields | Description |
|---|---|---|
| `assign_triage` | `patient_id`, `triage_category` | Assign a triage category to a patient |
| `admit_to_bed` | `patient_id`, `resource_id` | Admit a triaged patient to a resource |
| `discharge` | `patient_id` | Discharge a stable patient, freeing the resource |
| `order_test` | `patient_id`, `test_type` | Order a diagnostic test |
| `escalate` | `patient_id` | Immediately escalate patient to IMMEDIATE |
| `wait` | — | Do nothing this step |

### Triage Categories

| Category | Severity | Description |
|---|---|---|
| `immediate` | 1 | Life-threatening — treat NOW |
| `urgent` | 2 | Serious — treat within 30 min |
| `semi_urgent` | 3 | Moderate — treat within 1–2 hours |
| `non_urgent` | 4 | Minor — treat within 4–24 hours |

---

## 🏆 Tasks

### Task 1: `easy_triage` *(Difficulty: Easy)*

**Objective**: Correctly triage 5 patients with clear-cut presentations.

- 5 patients with obvious clinical presentations
- Abundant resources (no scarcity)
- Max 15 steps
- Expected agent score: **0.55–0.75**

---

### Task 2: `medium_triage` *(Difficulty: Medium)*

**Objective**: Triage and allocate resources for 10 patients with mixed presentations.

- 10 patients (2 critical, 2 serious, 3 moderate, 3 minor)
- Moderate resource constraints (2 ICU beds, 4 emergency beds)
- Must balance admission priority
- Max 35 steps
- Expected agent score: **0.35–0.55**

---

### Task 3: `hard_triage` *(Difficulty: Hard)*

**Objective**: Mass casualty scenario — maximize lives saved under extreme scarcity.

- 15 patients (3 critical, 4 serious, 4 moderate, 4 minor)
- Severe resource constraints (2 ICU beds, 3 emergency beds, 1 ventilator)
- Agent must make difficult trade-off decisions
- Max 50 steps
- Expected agent score: **0.20–0.40**

---

## 💰 Reward Function

Reward is computed **at every step**, providing dense partial progress signals:

| Component | Range | Signal |
|---|---|---|
| **Triage Accuracy** | -0.10 to +0.40 | Correct category = +0.40, off-by-one = +0.15, far off = -0.10 |
| **Wait Time Penalty** | -0.30 to 0.00 | Critical patients waiting >5 min accumulate penalties |
| **Resource Efficiency** | -0.15 to +0.30 | Matching severity to right resource type is rewarded |
| **Throughput Bonus** | -0.10 to +0.20 | Discharging appropriate patients frees capacity (+0.20) |
| **Deterioration Penalty** | -0.20 to 0.00 | Critical patients waiting >10 min receive deterioration penalty |

**Total per-step reward**: [-1.0, 1.0]

**Final episode score (grader)**: [0.0, 1.0] weighted across:
- 40% triage accuracy
- 30% critical care timeliness
- 20% resource utilization efficiency
- 10% throughput

---

## 🚀 Setup & Usage

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)

### Local Setup

```bash
# 1. Clone / download the repo
cd hospital-triage-env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the environment server
python server.py
# Server runs at http://localhost:7860
```

### Run Inference (Baseline Agent)

```bash
export HF_TOKEN="your_huggingface_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

python inference.py
```

### Docker

```bash
# Build
docker build -t hospital-triage-env .

# Run
docker run -p 7860:7860 hospital-triage-env

# Test
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy_triage", "seed": 42}'
```

---

## 🔗 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode. Body: `{task_name, seed, session_id}` |
| `/step` | POST | Take an action. Body: `{action_type, patient_id, triage_category, ...}` |
| `/state` | GET | Get full environment state. Query: `?session_id=default` |
| `/grade` | GET | Get final normalized score [0,1]. Query: `?session_id=default` |
| `/tasks` | GET | List all available tasks |
| `/health` | GET | Health check |

### Example Session

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -d '{"task_name":"easy_triage"}'

# Triage a patient
curl -X POST http://localhost:7860/step \
  -d '{"action_type":"assign_triage","patient_id":"P001","triage_category":"immediate"}'

# Admit patient to ICU
curl -X POST http://localhost:7860/step \
  -d '{"action_type":"admit_to_bed","patient_id":"P001","resource_id":"icu_bed_1"}'

# Get score
curl http://localhost:7860/grade
```

---

## 📊 Baseline Scores

Baseline agent: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router

| Task | Score | Steps | Success |
|---|---|---|---|
| easy_triage | ~0.62 | 10 | ✅ |
| medium_triage | ~0.44 | 28 | ✅ |
| hard_triage | ~0.31 | 45 | ❌ |

---

## 🏗 HuggingFace Spaces Deployment

1. Create a new HF Space (Docker SDK)
2. Push this repository to the Space
3. Set environment variables in Space settings:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`

The Space will build the Docker image and expose the API on port 7860.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
