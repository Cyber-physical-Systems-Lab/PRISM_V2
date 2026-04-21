"""
Event-triggered vs fixed-interval replanning for RQ3.

Research question:
  Can the CRPP-PEC update interval be made adaptive — triggered by energy
  deviation events rather than fixed time — to avoid the cycling failures
  identified in REACH-MP while retaining delay-reduction benefits?

Hypothesis:
  An event-triggered replanning policy based on energy state deviation from
  predicted threshold outperforms fixed-interval replanning in both safety and
  throughput, by avoiding premature replanning while responding faster to
  genuine energy deviations.

Key concepts:
  - TaskSnapshot    : Stage-1 energy prediction captured at task assignment.
  - Deviation       : predicted_battery(t) − actual_battery(t).
  - should_replan() : trigger gate (FIXED: interval elapsed; EVENT: deviation > δ).
  - check_feasibility_*() : decision gate after trigger fires.
  - record_decision(): logs trigger + decision for post-hoc stability analysis.
  - Oscillation     : within a W-step window the same agent receives both
                      "continue" and "abort_charge" — REACH-MP cycling analogue.
  - Stability δ*    : minimum δ s.t. oscillation_count(W) = 0 for EVENT policy.
                      Analogous to REACH-MP's minimum interval T_min (RQ3c).

Sub-questions:
  RQ3a: Sweep δ → find optimal deviation threshold without oscillation.
  RQ3b: stage2_agreement_rate quantifies how often weight-revelation rechecks
        agree with Stage-1 prediction.
  RQ3c: stability_index(W) vs δ or K identifies the stability equilibrium.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Policy enum
# ---------------------------------------------------------------------------

class ReplanningPolicy(Enum):
    NO_REPLANNING   = "none"    # Baseline: commit to mission until completion
    FIXED_INTERVAL  = "fixed"   # Replan every K steps (REACH-MP analog)
    EVENT_TRIGGERED = "event"   # Replan when energy deviation > delta


# ---------------------------------------------------------------------------
# Task snapshot — Stage-1 energy state at assignment time
# ---------------------------------------------------------------------------

@dataclass
class _TaskSnapshot:
    """
    Records Stage-1 prediction at task assignment so deviation can be computed
    at any later step:

        predicted_battery(t) = battery_0 − cost_rate × (t − t_0)
        deviation(t)         = predicted_battery(t) − actual_battery(t)

    Positive deviation ⟹ agent used MORE energy than predicted (warning signal).
    """
    agent_id: int
    assigned_step: int
    battery_at_assignment: float
    predicted_task_cost: float   # Stage-1 estimated total cost (J)
    predicted_steps: int         # Estimated total steps to complete task

    # Updated at PICKING → DELIVERING (Stage-2, weight revealed)
    phase: str = "picking"       # "picking" | "delivering"
    payload_kg: float = 0.0
    phase_transition_step: int = -1
    phase_transition_battery: float = 0.0

    # Trigger bookkeeping
    last_trigger_step: int = field(default=-1)

    @property
    def cost_rate(self) -> float:
        return self.predicted_task_cost / max(self.predicted_steps, 1)

    def predicted_battery_at(self, step: int) -> float:
        elapsed = max(0, step - self.assigned_step)
        return self.battery_at_assignment - elapsed * self.cost_rate

    def deviation_at(self, actual_battery: float, step: int) -> float:
        """Positive ⟹ consumed more than predicted."""
        return self.predicted_battery_at(step) - actual_battery


# ---------------------------------------------------------------------------
# Replan event record
# ---------------------------------------------------------------------------

@dataclass
class _ReplanRecord:
    step: int
    agent_id: int
    trigger: str      # "fixed_interval" | "energy_deviation" | "weight_revealed"
    decision: str     # "continue" | "abort_charge"
    battery_actual: float
    deviation: float
    mission_phase: str


# ---------------------------------------------------------------------------
# Replanning metrics
# ---------------------------------------------------------------------------

class ReplanningMetrics:
    """
    Per-episode replanning statistics returned by ReplanningController.get_metrics().

    Key attributes for each RQ3 sub-question:
      RQ3a — total_triggers, oscillation_count(W), stability_index(L, W)
      RQ3b — stage2_agreement_rate  (Stage-2 agrees with Stage-1 prediction)
      RQ3c — stability_index vs delta/interval sweep finds δ* / K_min
    """

    def __init__(self) -> None:
        self.missions_started: int = 0
        self.missions_completed: int = 0
        self.total_replans: int = 0     # trigger fires (all types)
        self.mission_aborts: int = 0    # "abort_charge" decisions
        self.safety_violations: int = 0 # battery=0 mid-task

        # Stage-2 recheck stats (RQ3b)
        self.stage2_checks: int = 0     # weight-revealed triggers
        self.stage2_aborts: int = 0     # stage-2 caused abort

        # Prediction error distribution (sampled per step per active AGV)
        self.prediction_errors: List[float] = []

        # Full event log for oscillation analysis
        self._records: List[_ReplanRecord] = []

    # -- recording ----------------------------------------------------------

    def _log(self, rec: _ReplanRecord) -> None:
        self._records.append(rec)
        self.total_replans += 1
        if rec.decision == "abort_charge":
            self.mission_aborts += 1
        if rec.trigger == "weight_revealed":
            self.stage2_checks += 1
            if rec.decision == "abort_charge":
                self.stage2_aborts += 1

    # -- derived metrics ----------------------------------------------------

    def stage2_agreement_rate(self) -> float:
        """Fraction of Stage-2 rechecks that agreed with Stage-1 (no abort)."""
        if self.stage2_checks == 0:
            return 1.0
        return (self.stage2_checks - self.stage2_aborts) / self.stage2_checks

    def abort_rate(self) -> float:
        return self.mission_aborts / max(self.missions_started, 1)

    def replan_rate(self, episode_length: int) -> float:
        return self.total_replans / max(episode_length, 1)

    def oscillation_count(self, window: int = 5) -> int:
        """
        Count of oscillation events.

        An oscillation event occurs when for the same agent, two consecutive
        replan records within `window` steps have opposite decisions
        (one "continue" and one "abort_charge").  This is the agent-level
        cycling analogue of the REACH-MP minimum-interval failure mode.
        """
        if len(self._records) < 2:
            return 0
        # Group events by agent_id, sorted by step
        by_agent: Dict[int, List[_ReplanRecord]] = {}
        for rec in self._records:
            by_agent.setdefault(rec.agent_id, []).append(rec)
        for recs in by_agent.values():
            recs.sort(key=lambda r: r.step)

        count = 0
        for recs in by_agent.values():
            for a, b in zip(recs, recs[1:]):
                if b.step - a.step <= window and a.decision != b.decision:
                    count += 1
        return count

    def stability_index(self, episode_length: int, window: int = 5) -> float:
        """
        Stability index: fraction of all consecutive replan pairs that are
        NOT oscillating.  Range [0, 1]:
          1.0 = perfectly stable (no oscillation)
          0.0 = every consecutive pair oscillates
        Analogous to REACH-MP's stability condition: T > T_min.
        For event-triggered, stability_index ≈ 1.0 iff δ > δ*.
        """
        if len(self._records) < 2:
            return 1.0
        by_agent: Dict[int, List[_ReplanRecord]] = {}
        for rec in self._records:
            by_agent.setdefault(rec.agent_id, []).append(rec)
        for recs in by_agent.values():
            recs.sort(key=lambda r: r.step)

        total_pairs = 0
        oscillating_pairs = 0
        for recs in by_agent.values():
            for a, b in zip(recs, recs[1:]):
                total_pairs += 1
                if b.step - a.step <= window and a.decision != b.decision:
                    oscillating_pairs += 1

        if total_pairs == 0:
            return 1.0
        return 1.0 - oscillating_pairs / total_pairs

    def is_stable(self, window: int = 5) -> bool:
        """True if oscillation_count is 0 for the given window."""
        return self.oscillation_count(window) == 0

    def mean_prediction_error(self) -> float:
        return float(np.mean(self.prediction_errors)) if self.prediction_errors else 0.0

    def std_prediction_error(self) -> float:
        return float(np.std(self.prediction_errors)) if self.prediction_errors else 0.0

    def summary(self, episode_length: int, window: int = 5) -> Dict:
        """
        Return a flat dict of all metrics suitable for JSON serialisation
        and episode-level aggregation in study_replanning.py.

        `oscillation_count` and `stability_index` are also included here
        using the provided `window`; callers can override them by re-calling
        the dedicated methods with a different window.
        """
        osc = self.oscillation_count(window)
        si  = self.stability_index(episode_length, window)
        return {
            "total_replans":         self.total_replans,
            "mission_aborts":        self.mission_aborts,
            "safety_violations":     self.safety_violations,
            "missions_started":      self.missions_started,
            "missions_completed":    self.missions_completed,
            "replan_rate":           self.replan_rate(episode_length),
            "abort_rate":            self.abort_rate(),
            "stage2_checks":         self.stage2_checks,
            "stage2_aborts":         self.stage2_aborts,
            "stage2_agreement_rate": self.stage2_agreement_rate(),
            "oscillation_count":     osc,
            "stability_index":       si,
            "is_stable":             osc == 0,
            "mean_prediction_error": self.mean_prediction_error(),
            "std_prediction_error":  self.std_prediction_error(),
        }


# ---------------------------------------------------------------------------
# Replanning controller
# ---------------------------------------------------------------------------

class ReplanningController:
    """
    Mid-task replanning controller for AGVs in a heuristic episode.

    Supports three policies (ReplanningPolicy):
      NO_REPLANNING   — never interrupt an in-progress task.
      FIXED_INTERVAL  — re-evaluate feasibility every `replan_interval` steps.
      EVENT_TRIGGERED — re-evaluate when battery deviation > `deviation_threshold`.

    Lifecycle (called from heuristic_episode):
    ──────────────────────────────────────────
    1. Task assigned (PICKING)   → on_mission_assigned(agent_id, step, battery,
                                                       phase="picking", payload_kg=0)
    2. Shelf picked up           → on_phase_transition(agent_id, step, battery,
                                                        new_phase="delivering",
                                                        payload_kg=actual_kg)
    3. Per step, per active AGV  → (a) should_replan(agent_id, step, battery)
                                   (b) if True: check_feasibility_picking/delivering(...)
                                   (c) record_decision(agent_id, step, battery,
                                                       decision, mission_phase, trigger)
    4. Task completed / aborted  → on_mission_complete(agent_id)
    5. Battery = 0 mid-task      → record_safety_violation()

    Stage-2 weight revelation (RQ3b, use_stage2_recheck=True):
      After PICKING → DELIVERING transition, caller directly calls
      check_feasibility_delivering() and record_decision(trigger="weight_revealed").
    """

    def __init__(
        self,
        policy: ReplanningPolicy,
        energy_model=None,
        replan_interval: int = 5,
        deviation_threshold: float = 8.0,
        use_stage2_recheck: bool = True,
    ) -> None:
        self.policy    = policy
        self.interval  = replan_interval
        self.delta     = deviation_threshold
        self.stage2    = use_stage2_recheck

        from tarware.energy_coupling import EnergyModel
        self.model = energy_model or EnergyModel()

        self._snapshots: Dict[int, _TaskSnapshot] = {}
        self._metrics   = ReplanningMetrics()

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def on_mission_assigned(
        self,
        agent_id: int,
        step: int,
        battery: float,
        phase: str = "picking",
        payload_kg: float = 0.0,
    ) -> None:
        """
        Record Stage-1 energy snapshot at task assignment time.
        Uses the package weight prior for predicted cost (conservative).
        """
        # Stage-1 predicted cost for a typical pick-deliver cycle
        # (prior mean weight, rough path estimate)
        typical_shelf_path = 15          # warehouse-scale rough estimate
        typical_goal_path  = 15
        cost_empty = self.model.agv_path_cost(typical_shelf_path)
        loaded_mu, _ = self.model.agv_path_cost_prior(typical_goal_path)
        predicted_total = cost_empty + loaded_mu + self.model.agv_load_cost * 2
        predicted_steps = typical_shelf_path + typical_goal_path + 4

        self._snapshots[agent_id] = _TaskSnapshot(
            agent_id=agent_id,
            assigned_step=step,
            battery_at_assignment=battery,
            predicted_task_cost=max(predicted_total, 0.1),
            predicted_steps=max(predicted_steps, 1),
            phase=phase,
            payload_kg=payload_kg,
            last_trigger_step=step,
        )
        self._metrics.missions_started += 1

    def on_phase_transition(
        self,
        agent_id: int,
        step: int,
        battery: float,
        new_phase: str,
        payload_kg: float,
    ) -> None:
        """
        Called at PICKING → DELIVERING when the shelf weight is revealed.
        Updates the snapshot's phase and payload for more accurate deviation tracking.
        """
        snap = self._snapshots.get(agent_id)
        if snap is None:
            return
        snap.phase = new_phase
        snap.payload_kg = payload_kg
        snap.phase_transition_step = step
        snap.phase_transition_battery = battery
        # Refine predicted cost: update remaining cost with revealed weight
        # (the battery already consumed in picking phase is factored out)
        typical_goal_path = 15
        snap.predicted_task_cost = (
            self.model.agv_path_cost(typical_goal_path, payload_kg)
            + self.model.agv_load_cost
        )
        snap.predicted_steps = typical_goal_path + 2
        snap.assigned_step = step
        snap.battery_at_assignment = battery

    def on_mission_complete(self, agent_id: int) -> None:
        """Called when task is done (successful delivery or aborted mid-task)."""
        self._snapshots.pop(agent_id, None)
        self._metrics.missions_completed += 1

    def record_safety_violation(self) -> None:
        """Battery = 0 while mid-task — safety failure."""
        self._metrics.safety_violations += 1

    # ------------------------------------------------------------------
    # Trigger gate
    # ------------------------------------------------------------------

    def should_replan(self, agent_id: int, step: int, battery: float) -> bool:
        """
        Check whether the replanning trigger fires for this agent at this step.

        Does NOT make the abort/continue decision; that is delegated to
        check_feasibility_picking() / check_feasibility_delivering().

        Side-effect: updates last_trigger_step when True (prevents immediate
        re-fire for FIXED_INTERVAL policy).

        Also records deviation for prediction-error analysis (NO_REPLANNING
        included — deviation still tracked for RQ3c post-hoc analysis).
        """
        snap = self._snapshots.get(agent_id)
        if snap is None:
            return False

        deviation = snap.deviation_at(battery, step)
        self._metrics.prediction_errors.append(deviation)

        if self.policy == ReplanningPolicy.NO_REPLANNING:
            return False

        if self.policy == ReplanningPolicy.FIXED_INTERVAL:
            if step - snap.last_trigger_step >= self.interval:
                snap.last_trigger_step = step
                return True
            return False

        if self.policy == ReplanningPolicy.EVENT_TRIGGERED:
            if deviation > self.delta:
                snap.last_trigger_step = step
                return True
            return False

        return False

    # ------------------------------------------------------------------
    # Decision gates (feasibility checks)
    # ------------------------------------------------------------------

    def check_feasibility_picking(
        self,
        battery: float,
        steps_to_shelf: int,
        steps_shelf_to_goal: int,
        steps_to_charger: int,
        alpha: float,
        low_battery_threshold: float,
    ) -> bool:
        """
        Stage-1 feasibility check for an AGV en route to pick up a shelf.
        Uses chance constraint with weight prior.

        Returns True if the agent can complete the task safely.
        """
        cost_empty = self.model.agv_path_cost(steps_to_shelf)
        loaded_mu, loaded_sigma = self.model.agv_path_cost_prior(steps_shelf_to_goal)
        task_cost = (cost_empty
                     + self.model.chance_constraint_threshold(loaded_mu, loaded_sigma, alpha)
                     + self.model.agv_load_cost * 2)
        charger_cost = self.model.agv_path_cost(steps_to_charger)
        return battery - task_cost - charger_cost >= low_battery_threshold

    def check_feasibility_delivering(
        self,
        battery: float,
        payload_kg: float,
        steps_to_goal: int,
        steps_to_charger: int,
        low_battery_threshold: float,
    ) -> bool:
        """
        Stage-2 feasibility check for an AGV already carrying a shelf
        (weight revealed → tight threshold, sigma → 0).

        Returns True if the agent can complete the delivery safely.
        """
        task_cost = (self.model.agv_path_cost(steps_to_goal, payload_kg)
                     + self.model.agv_load_cost)
        charger_cost = self.model.agv_path_cost(steps_to_charger)
        return battery - task_cost - charger_cost >= low_battery_threshold

    # ------------------------------------------------------------------
    # Decision logging
    # ------------------------------------------------------------------

    def record_decision(
        self,
        agent_id: int,
        step: int,
        battery_actual: float,
        decision: str,
        mission_phase: str,
        trigger: Optional[str] = None,
    ) -> None:
        """
        Log a replanning decision after a trigger fired.

        trigger (optional): one of "fixed_interval", "energy_deviation",
        "weight_revealed".  If None, inferred from the current policy.
        """
        if trigger is None:
            trigger = {
                ReplanningPolicy.FIXED_INTERVAL:  "fixed_interval",
                ReplanningPolicy.EVENT_TRIGGERED: "energy_deviation",
            }.get(self.policy, "energy_deviation")

        snap = self._snapshots.get(agent_id)
        deviation = snap.deviation_at(battery_actual, step) if snap else 0.0

        rec = _ReplanRecord(
            step=step,
            agent_id=agent_id,
            trigger=trigger,
            decision=decision,
            battery_actual=battery_actual,
            deviation=deviation,
            mission_phase=mission_phase,
        )
        self._metrics._log(rec)

    # ------------------------------------------------------------------
    # Metrics extraction
    # ------------------------------------------------------------------

    def get_metrics(self) -> ReplanningMetrics:
        return self._metrics
