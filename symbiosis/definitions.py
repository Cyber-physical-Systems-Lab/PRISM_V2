"""
C1 — Formal Definition of Robotic Symbiosis (Step 1.2)

A relationship R_{ij} between agents i and j is classified by
comparing fitness WITH and WITHOUT the relationship:

  W_i^R = fitness of i WITH relationship
  W_i^∅ = fitness of i WITHOUT relationship

  Mutualism:    W_i^R > W_i^∅  AND  W_j^R > W_j^∅   (+/+)
  Commensalism: W_i^R > W_i^∅  AND  W_j^R = W_j^∅   (+/0)
  Parasitism:   W_i^R > W_i^∅  AND  W_j^R < W_j^∅   (+/-)
  Competition:  W_i^R < W_i^∅  AND  W_j^R < W_j^∅   (-/-)
  Neutralism:   W_i^R = W_i^∅  AND  W_j^R = W_j^∅   (0/0)

"Without relationship" is operationalised as counterfactual:
what would fitness be if agents could not observe each other?
This is verifiable from the value function (see counterfactual_critic.py).
"""

from enum import Enum
from dataclasses import dataclass


class RelationshipType(Enum):
    MUTUALISM    = "mutualism"      # (+/+)
    COMMENSALISM = "commensalism"   # (+/0)
    PARASITISM   = "parasitism"     # (+/-)
    COMPETITION  = "competition"    # (-/-)
    NEUTRALISM   = "neutralism"     # (0/0)


@dataclass
class RelationshipMeasurement:
    agent_i: int
    agent_j: int
    W_i_with: float       # fitness of i WITH relationship
    W_i_without: float    # fitness of i WITHOUT relationship
    W_j_with: float
    W_j_without: float
    relationship_type: RelationshipType
    delta_i: float        # W_i^R - W_i^∅
    delta_j: float        # W_j^R - W_j^∅


class RoboticSymbiosisClassifier:
    """
    Classifies the interaction type between two agents using
    fitness measurements (or value-function counterfactuals).

    This is the formal operationalisation of C1.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        measurement_window: int = 500,
    ):
        """
        threshold : minimum fitness delta to count as benefit/cost.
                    Avoids spurious classification from noise.
        """
        self.threshold = threshold
        self.window = measurement_window

        self.fitness_with: dict    = {}
        self.fitness_without: dict = {}

    # ------------------------------------------------------------------
    # Primary classification method
    # ------------------------------------------------------------------

    def classify(
        self,
        agent_i: int,
        agent_j: int,
        W_i_with: float,
        W_j_with: float,
        W_i_without: float,
        W_j_without: float,
    ) -> RelationshipMeasurement:
        """
        Classify a relationship from fitness measurements.
        Core of the C1 formal contribution.
        """
        delta_i = W_i_with - W_i_without
        delta_j = W_j_with - W_j_without

        i_pos = delta_i >  self.threshold
        i_neg = delta_i < -self.threshold
        j_pos = delta_j >  self.threshold
        j_neg = delta_j < -self.threshold

        if i_pos and j_pos:
            rtype = RelationshipType.MUTUALISM
        elif i_pos and j_neg:
            rtype = RelationshipType.PARASITISM
        elif i_neg and j_neg:
            rtype = RelationshipType.COMPETITION
        elif (i_pos and not j_neg) or (j_pos and not i_neg):
            rtype = RelationshipType.COMMENSALISM
        else:
            rtype = RelationshipType.NEUTRALISM

        return RelationshipMeasurement(
            agent_i=agent_i,
            agent_j=agent_j,
            W_i_with=W_i_with,
            W_i_without=W_i_without,
            W_j_with=W_j_with,
            W_j_without=W_j_without,
            relationship_type=rtype,
            delta_i=delta_i,
            delta_j=delta_j,
        )

    # ------------------------------------------------------------------
    # Value-function counterfactual (used during online training)
    # ------------------------------------------------------------------

    def classify_from_value_functions(
        self,
        agent_i,
        agent_j,
        state,
        critic_joint,
        critic_marginal,
    ) -> RelationshipMeasurement:
        """
        Estimate fitness WITH and WITHOUT the relationship directly
        from learned value functions.

        W_i^R  ≈ V_i(s ; π_joint)
        W_i^∅  ≈ V_i(s ; π_i_only)  [partner zeroed in obs]

        This makes C1 verifiable during training, not just post-hoc.
        """
        W_i_with  = critic_joint.get_value(state, agent_id=agent_i.id)
        W_j_with  = critic_joint.get_value(state, agent_id=agent_j.id)

        W_i_without = critic_marginal.get_value(
            state, agent_id=agent_i.id, exclude_agent=agent_j.id
        )
        W_j_without = critic_marginal.get_value(
            state, agent_id=agent_j.id, exclude_agent=agent_i.id
        )

        return self.classify(
            agent_i.id, agent_j.id,
            W_i_with, W_j_with,
            W_i_without, W_j_without,
        )
