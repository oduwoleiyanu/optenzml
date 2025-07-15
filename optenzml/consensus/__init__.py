"""
Consensus module for OptEnzML

This module contains the consensus prediction system that combines outputs
from multiple individual predictors using machine learning.
"""

from .consensus_model import ConsensusModel, ConsensusResult
# from .train_consensus import ConsensusTrainer  # Optional training module

__all__ = [
    "ConsensusModel",
    "ConsensusResult"
]
