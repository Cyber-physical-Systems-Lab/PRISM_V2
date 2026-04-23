"""
Robotic Symbiosis — core theoretical framework.

C1: Formal definition of robotic symbiosis from value functions.
C2: Reward decomposition with convergence guarantees.
C3: Experimental validation — symbiotic vs flat-cooperative reward conditions.
"""

from symbiosis.fitness import AgentFitness
from symbiosis.definitions import RelationshipType, RelationshipMeasurement, RoboticSymbiosisClassifier
from symbiosis.counterfactual_critic import JointCritic, MarginalCritic
from symbiosis.reward_decomposition import SymbioticRewardDecomposer, RewardDecomposition
from symbiosis.convergence import ConvergenceMonitor

__all__ = [
    "AgentFitness",
    "RelationshipType",
    "RelationshipMeasurement",
    "RoboticSymbiosisClassifier",
    "JointCritic",
    "MarginalCritic",
    "SymbioticRewardDecomposer",
    "RewardDecomposition",
    "ConvergenceMonitor",
]
