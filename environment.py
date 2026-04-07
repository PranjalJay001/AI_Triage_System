"""
Hospital Triage OpenEnv Environment
Real-world simulation of emergency department patient triage using RL.
Implements full OpenEnv spec: typed models, step/reset/state API.
"""

import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
#  Domain Enumerations
# ─────────────────────────────────────────────

class TriageCategory(str, Enum):
    IMMEDIATE = "immediate"      # Life-threatening, treat NOW
    URGENT = "urgent"            # Serious, treat within 30 min
    SEMI_URGENT = "semi_urgent"  # Treat within 1–2 hours
    NON_URGENT = "non_urgent"    # Treat within 4–24 hours
    DECEASED = "deceased"        # Do not resuscitate


class ActionType(str, Enum):
    ASSIGN_TRIAGE = "assign_triage"
    ADMIT_TO_BED = "admit_to_bed"
    DISCHARGE = "discharge"
    ORDER_TEST = "order_test"
    ESCALATE = "escalate"
    WAIT = "wait"


class ResourceType(str, Enum):
    ICU_BED = "icu_bed"
    EMERGENCY_BED = "emergency_bed"
    GENERAL_BED = "general_bed"
    VENTILATOR = "ventilator"
    XRAY = "xray"
    LAB = "lab"


# ─────────────────────────────────────────────
#  Pydantic Models (OpenEnv Typed Models)
# ─────────────────────────────────────────────

class Patient(BaseModel):
    patient_id: str
    age: int
    chief_complaint: str
    vital_signs: Dict[str, float]  # heart_rate, bp_systolic, spo2, temp, resp_rate
    pain_score: int                # 0–10
    arrival_time: float
    true_severity: int             # 1=critical, 2=serious, 3=moderate, 4=minor (hidden from agent)
    assigned_triage: Optional[TriageCategory] = None
    assigned_bed: Optional[str] = None
    tests_ordered: List[str] = Field(default_factory=list)
    wait_time: float = 0.0
    is_admitted: bool = False
    is_discharged: bool = False
    deterioration_risk: float = 0.0  # 0–1, increases with wait for critical patients


class Resource(BaseModel):
    resource_id: str
    resource_type: ResourceType
    is_occupied: bool = False
    patient_id: Optional[str] = None
    occupied_since: Optional[float] = None


class TriageObservation(BaseModel):
    """What the agent observes at each step."""
    current_time: float
    waiting_patients: List[Patient]
    admitted_patients: List[Patient]
    available_resources: Dict[str, int]  # resource_type -> count available
    current_patient: Optional[Patient]   # patient currently being triaged
    last_action_result: str
    last_action_error: Optional[str]
    queue_length: int
    avg_wait_time: float
    critical_patients_waiting: int
    step_number: int
    task_name: str


class TriageAction(BaseModel):
    """Action the agent can take."""
    action_type: ActionType
    patient_id: Optional[str] = None
    triage_category: Optional[TriageCategory] = None
    resource_id: Optional[str] = None
    test_type: Optional[str] = None
    notes: Optional[str] = None


class TriageReward(BaseModel):
    """Reward breakdown for interpretability."""
    total: float
    triage_accuracy: float       # Was severity correctly identified?
    wait_time_penalty: float     # Penalty for critical patients waiting
    resource_efficiency: float   # Bonus for optimal resource use
    throughput_bonus: float      # Bonus for processing patients
    deterioration_penalty: float # Penalty if patient deteriorated
    explanation: str


class StepResult(BaseModel):
    observation: TriageObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResult(BaseModel):
    observation: TriageObservation
    info: Dict[str, Any]


# ─────────────────────────────────────────────
#  Patient Generator
# ─────────────────────────────────────────────

CHIEF_COMPLAINTS = {
    1: [  # Critical
        "chest pain with ST elevation",
        "respiratory failure, SpO2 82%",
        "unresponsive, GCS 3",
        "massive hemorrhage",
        "anaphylactic shock",
        "stroke with facial droop",
    ],
    2: [  # Serious
        "severe chest pain, diaphoretic",
        "difficulty breathing, SpO2 91%",
        "altered mental status",
        "suspected sepsis, fever 39.8°C",
        "severe abdominal pain",
        "suspected fracture with neurovascular compromise",
    ],
    3: [  # Moderate
        "moderate chest pain, stable vitals",
        "asthma exacerbation, mild",
        "laceration requiring sutures",
        "urinary tract infection with fever",
        "hypertensive urgency",
        "suspected fracture, neurovascularly intact",
    ],
    4: [  # Minor
        "sore throat for 3 days",
        "ankle sprain, weight-bearing",
        "minor laceration, controlled bleeding",
        "ear pain",
        "back pain, chronic",
        "prescription refill request",
    ],
}

SEVERITY_TO_TRIAGE = {
    1: TriageCategory.IMMEDIATE,
    2: TriageCategory.URGENT,
    3: TriageCategory.SEMI_URGENT,
    4: TriageCategory.NON_URGENT,
}


def generate_patient(patient_id: str, severity: Optional[int] = None, current_time: float = 0.0) -> Patient:
    if severity is None:
        severity = random.choices([1, 2, 3, 4], weights=[15, 25, 35, 25])[0]

    complaint = random.choice(CHIEF_COMPLAINTS[severity])

    # Generate vital signs consistent with severity
    if severity == 1:
        heart_rate = random.uniform(40, 60) if random.random() < 0.3 else random.uniform(120, 160)
        bp_systolic = random.uniform(60, 90) if random.random() < 0.5 else random.uniform(180, 220)
        spo2 = random.uniform(78, 88)
        temp = random.uniform(35.0, 36.0) if random.random() < 0.3 else random.uniform(39.5, 41.0)
        resp_rate = random.uniform(6, 10) if random.random() < 0.3 else random.uniform(28, 40)
        pain_score = random.randint(7, 10)
    elif severity == 2:
        heart_rate = random.uniform(100, 130)
        bp_systolic = random.uniform(90, 110) if random.random() < 0.3 else random.uniform(160, 185)
        spo2 = random.uniform(88, 93)
        temp = random.uniform(38.5, 40.0)
        resp_rate = random.uniform(20, 28)
        pain_score = random.randint(5, 8)
    elif severity == 3:
        heart_rate = random.uniform(85, 110)
        bp_systolic = random.uniform(130, 160)
        spo2 = random.uniform(93, 97)
        temp = random.uniform(37.5, 38.8)
        resp_rate = random.uniform(16, 22)
        pain_score = random.randint(3, 6)
    else:
        heart_rate = random.uniform(65, 90)
        bp_systolic = random.uniform(110, 135)
        spo2 = random.uniform(97, 100)
        temp = random.uniform(36.5, 37.5)
        resp_rate = random.uniform(14, 18)
        pain_score = random.randint(1, 4)

    deterioration_risk = [0.8, 0.4, 0.1, 0.02][severity - 1]

    return Patient(
        patient_id=patient_id,
        age=random.randint(1, 95),
        chief_complaint=complaint,
        vital_signs={
            "heart_rate": round(heart_rate, 1),
            "bp_systolic": round(bp_systolic, 1),
            "spo2": round(spo2, 1),
            "temperature": round(temp, 1),
            "resp_rate": round(resp_rate, 1),
        },
        pain_score=pain_score,
        arrival_time=current_time,
        true_severity=severity,
        deterioration_risk=deterioration_risk,
    )


# ─────────────────────────────────────────────
#  Task Definitions
# ─────────────────────────────────────────────

TASKS = {
    "easy_triage": {
        "description": "Correctly triage 5 patients with clear-cut presentations. No resource allocation needed.",
        "difficulty": "easy",
        "n_patients": 5,
        "max_steps": 15,
        "severities": [1, 2, 3, 4, 4],  # fixed for reproducibility
        "resource_config": {"icu_bed": 5, "emergency_bed": 10, "general_bed": 20, "ventilator": 5, "xray": 3, "lab": 5},
    },
    "medium_triage": {
        "description": "Triage and allocate resources for 10 patients with mixed presentations. Must balance ICU/emergency beds.",
        "difficulty": "medium",
        "n_patients": 10,
        "max_steps": 35,
        "severities": [1, 1, 2, 2, 3, 3, 3, 4, 4, 4],
        "resource_config": {"icu_bed": 2, "emergency_bed": 4, "general_bed": 8, "ventilator": 2, "xray": 2, "lab": 3},
    },
    "hard_triage": {
        "description": "Mass casualty scenario: triage 15 patients under resource scarcity. Maximize lives saved with minimal resources.",
        "difficulty": "hard",
        "n_patients": 15,
        "max_steps": 50,
        "severities": [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
        "resource_config": {"icu_bed": 2, "emergency_bed": 3, "general_bed": 5, "ventilator": 1, "xray": 1, "lab": 2},
    },
}


# ─────────────────────────────────────────────
#  Hospital Triage Environment
# ─────────────────────────────────────────────

class HospitalTriageEnv:
    """
    OpenEnv-compliant Hospital Triage environment.

    An AI agent acts as an emergency department triage officer.
    It receives patient presentations and must:
      1. Assign correct triage categories (IMMEDIATE/URGENT/SEMI_URGENT/NON_URGENT)
      2. Allocate scarce resources (ICU beds, ventilators, etc.)
      3. Discharge stable patients to free capacity
      4. Minimize wait times for critical patients

    Reward is shaped to provide partial progress signals throughout the episode.
    """

    def __init__(self, task_name: str = "easy_triage", seed: int = 42):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASKS.keys())}")
        self.task_name = task_name
        self.task_config = TASKS[task_name]
        self.seed = seed
        self._rng = random.Random(seed)
        random.seed(seed)

        # State
        self._patients: Dict[str, Patient] = {}
        self._resources: Dict[str, Resource] = {}
        self._waiting_queue: List[str] = []
        self._admitted: List[str] = []
        self._discharged: List[str] = []
        self._current_time: float = 0.0
        self._step_count: int = 0
        self._done: bool = False
        self._last_action_result: str = "Episode not started"
        self._last_action_error: Optional[str] = None
        self._episode_rewards: List[float] = []
        self._triage_decisions: List[Tuple[str, TriageCategory, int]] = []  # (pid, assigned, true)
        self._patient_counter: int = 0

    # ── Internal helpers ──

    def _init_resources(self):
        self._resources = {}
        cfg = self.task_config["resource_config"]
        for rtype, count in cfg.items():
            for i in range(count):
                rid = f"{rtype}_{i+1}"
                self._resources[rid] = Resource(
                    resource_id=rid,
                    resource_type=ResourceType(rtype),
                )

    def _available_resource_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self._resources.values():
            if not r.is_occupied:
                counts[r.resource_type.value] = counts.get(r.resource_type.value, 0) + 1
        return counts

    def _find_free_resource(self, rtype: ResourceType) -> Optional[Resource]:
        for r in self._resources.values():
            if r.resource_type == rtype and not r.is_occupied:
                return r
        return None

    def _get_observation(self) -> TriageObservation:
        waiting = [self._patients[pid] for pid in self._waiting_queue]
        admitted = [self._patients[pid] for pid in self._admitted]

        # compute avg wait
        waits = [self._current_time - p.arrival_time for p in waiting]
        avg_wait = sum(waits) / len(waits) if waits else 0.0

        # critical patients waiting (not yet triaged as IMMEDIATE)
        critical_waiting = sum(
            1 for p in waiting
            if p.true_severity == 1 and not p.is_admitted
        )

        # current patient = first untriaged in queue
        current = None
        for pid in self._waiting_queue:
            p = self._patients[pid]
            if p.assigned_triage is None:
                current = p
                break

        return TriageObservation(
            current_time=self._current_time,
            waiting_patients=waiting,
            admitted_patients=admitted,
            available_resources=self._available_resource_counts(),
            current_patient=current,
            last_action_result=self._last_action_result,
            last_action_error=self._last_action_error,
            queue_length=len(self._waiting_queue),
            avg_wait_time=avg_wait,
            critical_patients_waiting=critical_waiting,
            step_number=self._step_count,
            task_name=self.task_name,
        )

    def _compute_reward(self, action: TriageAction) -> TriageReward:
        triage_acc = 0.0
        wait_penalty = 0.0
        resource_eff = 0.0
        throughput_bonus = 0.0
        deterioration_penalty = 0.0

        # 1. Triage accuracy reward
        if action.action_type == ActionType.ASSIGN_TRIAGE and action.patient_id and action.triage_category:
            p = self._patients.get(action.patient_id)
            if p:
                correct_cat = SEVERITY_TO_TRIAGE[p.true_severity]
                if action.triage_category == correct_cat:
                    triage_acc = 0.4
                elif abs(list(TriageCategory).index(action.triage_category) -
                         list(TriageCategory).index(correct_cat)) == 1:
                    triage_acc = 0.15  # off by one category — partial credit
                else:
                    triage_acc = -0.1  # significantly wrong

        # 2. Wait time penalty for critical patients
        for pid in self._waiting_queue:
            p = self._patients[pid]
            wait = self._current_time - p.arrival_time
            if p.true_severity == 1 and wait > 5.0:
                wait_penalty -= min(0.3, 0.05 * (wait - 5.0))
            elif p.true_severity == 2 and wait > 15.0:
                wait_penalty -= min(0.15, 0.02 * (wait - 15.0))

        # 3. Resource efficiency
        if action.action_type == ActionType.ADMIT_TO_BED and action.patient_id and action.resource_id:
            p = self._patients.get(action.patient_id)
            r = self._resources.get(action.resource_id)
            if p and r:
                # reward matching severity to right resource type
                ideal = {1: ResourceType.ICU_BED, 2: ResourceType.EMERGENCY_BED,
                         3: ResourceType.EMERGENCY_BED, 4: ResourceType.GENERAL_BED}
                if r.resource_type == ideal.get(p.true_severity):
                    resource_eff = 0.3
                elif r.resource_type == ResourceType.ICU_BED and p.true_severity > 2:
                    resource_eff = -0.15  # wasteful — using ICU for minor
                else:
                    resource_eff = 0.1   # suboptimal but not terrible

        # 4. Throughput bonus for discharge
        if action.action_type == ActionType.DISCHARGE and action.patient_id:
            p = self._patients.get(action.patient_id)
            if p and p.is_discharged:
                if p.true_severity >= 3:
                    throughput_bonus = 0.2   # good — freed capacity
                else:
                    throughput_bonus = -0.1  # bad — discharged critical patient

        # 5. Deterioration penalty
        for pid in self._waiting_queue:
            p = self._patients[pid]
            wait = self._current_time - p.arrival_time
            if p.true_severity == 1 and wait > 10.0:
                deterioration_penalty -= 0.2

        total = triage_acc + wait_penalty + resource_eff + throughput_bonus + deterioration_penalty
        total = max(-1.0, min(1.0, total))

        parts = []
        if triage_acc != 0: parts.append(f"triage={triage_acc:+.2f}")
        if wait_penalty != 0: parts.append(f"wait_pen={wait_penalty:+.2f}")
        if resource_eff != 0: parts.append(f"resource={resource_eff:+.2f}")
        if throughput_bonus != 0: parts.append(f"throughput={throughput_bonus:+.2f}")
        if deterioration_penalty != 0: parts.append(f"deterioration={deterioration_penalty:+.2f}")

        return TriageReward(
            total=total,
            triage_accuracy=triage_acc,
            wait_time_penalty=wait_penalty,
            resource_efficiency=resource_eff,
            throughput_bonus=throughput_bonus,
            deterioration_penalty=deterioration_penalty,
            explanation=" | ".join(parts) if parts else "no significant action",
        )

    def _is_done(self) -> bool:
        if self._step_count >= self.task_config["max_steps"]:
            return True
        # Episode ends when all patients are admitted or discharged
        all_processed = all(
            self._patients[pid].is_admitted or self._patients[pid].is_discharged
            for pid in self._patients
        )
        return all_processed

    def _spawn_patients(self):
        severities = self.task_config["severities"]
        for i, sev in enumerate(severities):
            pid = f"P{i+1:03d}"
            p = generate_patient(pid, sev, current_time=float(i * 2))  # staggered arrivals
            self._patients[pid] = p
            self._waiting_queue.append(pid)

    # ── OpenEnv Public API ──

    def reset(self) -> ResetResult:
        random.seed(self.seed)
        self._patients = {}
        self._resources = {}
        self._waiting_queue = []
        self._admitted = []
        self._discharged = []
        self._current_time = 0.0
        self._step_count = 0
        self._done = False
        self._last_action_result = "Episode started. Patients are waiting."
        self._last_action_error = None
        self._episode_rewards = []
        self._triage_decisions = []

        self._init_resources()
        self._spawn_patients()

        return ResetResult(
            observation=self._get_observation(),
            info={"task": self.task_name, "n_patients": len(self._patients),
                  "max_steps": self.task_config["max_steps"]},
        )

    def step(self, action: TriageAction) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        self._current_time += 1.0  # each step = 1 minute
        self._last_action_error = None

        # ── Process action ──
        if action.action_type == ActionType.ASSIGN_TRIAGE:
            if not action.patient_id or not action.triage_category:
                self._last_action_error = "assign_triage requires patient_id and triage_category"
            elif action.patient_id not in self._patients:
                self._last_action_error = f"Unknown patient: {action.patient_id}"
            else:
                p = self._patients[action.patient_id]
                p.assigned_triage = action.triage_category
                self._triage_decisions.append((action.patient_id, action.triage_category, p.true_severity))
                self._last_action_result = f"Assigned {action.patient_id} triage: {action.triage_category.value}"

        elif action.action_type == ActionType.ADMIT_TO_BED:
            if not action.patient_id or not action.resource_id:
                self._last_action_error = "admit_to_bed requires patient_id and resource_id"
            elif action.patient_id not in self._patients:
                self._last_action_error = f"Unknown patient: {action.patient_id}"
            elif action.resource_id not in self._resources:
                self._last_action_error = f"Unknown resource: {action.resource_id}"
            else:
                p = self._patients[action.patient_id]
                r = self._resources[action.resource_id]
                if r.is_occupied:
                    self._last_action_error = f"Resource {action.resource_id} is already occupied"
                elif p.assigned_triage is None:
                    self._last_action_error = "Patient must be triaged before admission"
                else:
                    r.is_occupied = True
                    r.patient_id = action.patient_id
                    r.occupied_since = self._current_time
                    p.assigned_bed = action.resource_id
                    p.is_admitted = True
                    if action.patient_id in self._waiting_queue:
                        self._waiting_queue.remove(action.patient_id)
                    if action.patient_id not in self._admitted:
                        self._admitted.append(action.patient_id)
                    self._last_action_result = f"Admitted {action.patient_id} to {action.resource_id}"

        elif action.action_type == ActionType.DISCHARGE:
            if not action.patient_id:
                self._last_action_error = "discharge requires patient_id"
            elif action.patient_id not in self._patients:
                self._last_action_error = f"Unknown patient: {action.patient_id}"
            else:
                p = self._patients[action.patient_id]
                p.is_discharged = True
                # free up bed if occupied
                if p.assigned_bed and p.assigned_bed in self._resources:
                    r = self._resources[p.assigned_bed]
                    r.is_occupied = False
                    r.patient_id = None
                if action.patient_id in self._waiting_queue:
                    self._waiting_queue.remove(action.patient_id)
                if action.patient_id in self._admitted:
                    self._admitted.remove(action.patient_id)
                if action.patient_id not in self._discharged:
                    self._discharged.append(action.patient_id)
                self._last_action_result = f"Discharged {action.patient_id}"

        elif action.action_type == ActionType.ORDER_TEST:
            if not action.patient_id or not action.test_type:
                self._last_action_error = "order_test requires patient_id and test_type"
            else:
                p = self._patients.get(action.patient_id)
                if p:
                    p.tests_ordered.append(action.test_type)
                    self._last_action_result = f"Ordered {action.test_type} for {action.patient_id}"

        elif action.action_type == ActionType.ESCALATE:
            if not action.patient_id:
                self._last_action_error = "escalate requires patient_id"
            else:
                p = self._patients.get(action.patient_id)
                if p:
                    p.assigned_triage = TriageCategory.IMMEDIATE
                    self._last_action_result = f"Escalated {action.patient_id} to IMMEDIATE"

        elif action.action_type == ActionType.WAIT:
            self._last_action_result = "Agent chose to wait (no action taken)"

        # Update wait times
        for pid in self._waiting_queue:
            p = self._patients[pid]
            p.wait_time = self._current_time - p.arrival_time

        reward_breakdown = self._compute_reward(action)
        reward = reward_breakdown.total
        self._episode_rewards.append(reward)
        self._done = self._is_done()

        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=self._done,
            info={
                "reward_breakdown": reward_breakdown.model_dump(),
                "step": self._step_count,
                "discharged": len(self._discharged),
                "admitted": len(self._admitted),
                "waiting": len(self._waiting_queue),
            },
        )

    def state(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "current_time": self._current_time,
            "step_count": self._step_count,
            "done": self._done,
            "patients": {pid: p.model_dump() for pid, p in self._patients.items()},
            "resources": {rid: r.model_dump() for rid, r in self._resources.items()},
            "waiting_queue": self._waiting_queue,
            "admitted": self._admitted,
            "discharged": self._discharged,
            "episode_rewards": self._episode_rewards,
        }

    def close(self):
        """Clean up resources."""
        pass

    # ── Grader ──

    def grade(self) -> float:
        """
        Compute final normalized score for the episode [0.0, 1.0].

        Scoring criteria:
          - Triage accuracy (40%): correct category assignment
          - Critical care timeliness (30%): IMMEDIATE patients admitted quickly
          - Resource utilization (20%): efficient use of limited resources
          - Throughput (10%): patients processed
        """
        if not self._patients:
            return 0.0

        n_patients = len(self._patients)

        # 1. Triage accuracy
        correct = 0
        partial = 0
        for pid, assigned_cat, true_sev in self._triage_decisions:
            correct_cat = SEVERITY_TO_TRIAGE[true_sev]
            if assigned_cat == correct_cat:
                correct += 1
            elif abs(list(TriageCategory).index(assigned_cat) -
                     list(TriageCategory).index(correct_cat)) == 1:
                partial += 1
        triaged_count = len(self._triage_decisions)
        if triaged_count > 0:
            triage_score = (correct + 0.5 * partial) / triaged_count
        else:
            triage_score = 0.0

        # 2. Critical care timeliness
        critical_patients = [p for p in self._patients.values() if p.true_severity == 1]
        if critical_patients:
            timely = sum(1 for p in critical_patients if p.is_admitted and p.wait_time <= 10.0)
            timeliness_score = timely / len(critical_patients)
        else:
            timeliness_score = 1.0

        # 3. Resource utilization (avoid wasting ICU on non-critical)
        total_resources = len(self._resources)
        good_alloc = 0
        for p in self._patients.values():
            if p.assigned_bed:
                r = self._resources.get(p.assigned_bed)
                if r:
                    ideal = {1: ResourceType.ICU_BED, 2: ResourceType.EMERGENCY_BED,
                             3: ResourceType.EMERGENCY_BED, 4: ResourceType.GENERAL_BED}
                    if r.resource_type == ideal.get(p.true_severity):
                        good_alloc += 1
                    elif r.resource_type != ResourceType.ICU_BED or p.true_severity <= 2:
                        good_alloc += 0.5
        admitted_count = len(self._admitted) + len(self._discharged)
        resource_score = good_alloc / max(admitted_count, 1) if admitted_count > 0 else 0.5

        # 4. Throughput
        throughput_score = (len(self._admitted) + len(self._discharged)) / n_patients

        # Weighted final score
        final = (
            0.40 * triage_score
            + 0.30 * timeliness_score
            + 0.20 * resource_score
            + 0.10 * throughput_score
        )
        return round(min(max(final, 0.0), 1.0), 4)
