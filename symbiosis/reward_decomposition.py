"""
C2 — Symbiotic Reward Decomposition (Step 2.1)

Total reward decomposes as:

  r_i(t) = r_i^task(t) + r_i^sym(t)

where:
  r_i^task(t) = individual task reward (deliveries, efficiency)

  r_i^sym(t)  = Σ_j  φ(type_{ij}(t)) × |δ_{ij}(t)|

  type_{ij}(t) = classified relationship type  (from C1)
  δ_{ij}(t)   = magnitude of fitness delta
  φ(·)        = type-dependent weight function:

    φ(MUTUALISM)    > 0   reward cooperation
    φ(COMMENSALISM) > 0   reward one-sided help
    φ(COMPETITION)  < 0   penalise mutual interference
    φ(PARASITISM)   < 0   penalise exploitation
    φ(NEUTRALISM)   = 0   no effect

Key design: mutualism becomes the dominant strategy when
capability complementarity (picker↔AGV) exists in the env.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from symbiosis.definitions import RelationshipType, RelationshipMeasurement


@dataclass
class RewardDecomposition:
    agent_id: int
    r_task: float
    r_sym: float
    r_total: float
    relationship_contributions: Dict[int, float] = field(default_factory=dict)
    # {partner_id: contribution to r_sym}


class SymbioticRewardDecomposer:
    """
    Implements the C2 reward decomposition.

    Separates task reward from the symbiotic bonus and
    conditions the bonus on classified interaction type.
    """

    def __init__(
        self,
        phi_mutualism:    float =  2.0,
        phi_commensalism: float =  1.0,
        phi_competition:  float = -1.0,
        phi_parasitism:   float = -0.5,
        phi_neutralism:   float =  0.0,
        sym_scale:        float =  0.3,
    ):
        """
        phi_* : φ(type) weight for each relationship type.
        sym_scale : prevents symbiotic component dominating task reward.
        """
        self.phi = {
            RelationshipType.MUTUALISM:    phi_mutualism,
            RelationshipType.COMMENSALISM: phi_commensalism,
            RelationshipType.COMPETITION:  phi_competition,
            RelationshipType.PARASITISM:   phi_parasitism,
            RelationshipType.NEUTRALISM:   phi_neutralism,
        }
        self.sym_scale = sym_scale

    def decompose(
        self,
        agent_id: int,
        task_reward: float,
        partner_measurements: List[RelationshipMeasurement],
    ) -> RewardDecomposition:
        """
        Compute full reward decomposition for one agent.

        r_i = r_i^task + sym_scale × Σ_j φ(type_ij) × |δ_ij|
        """
        r_sym = 0.0
        contributions: Dict[int, float] = {}

        for m in partner_measurements:
            if m.agent_i == agent_id:
                delta   = m.delta_i
                partner = m.agent_j
            else:
                delta   = m.delta_j
                partner = m.agent_i

            phi_val      = self.phi[m.relationship_type]
            contribution = phi_val * abs(delta)
            r_sym       += contribution
            contributions[partner] = contribution

        r_sym *= self.sym_scale

        return RewardDecomposition(
            agent_id=agent_id,
            r_task=task_reward,
            r_sym=r_sym,
            r_total=task_reward + r_sym,
            relationship_contributions=contributions,
        )
