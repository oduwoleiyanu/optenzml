"""
OptEnzML - Enzyme Optimal Temperature Prediction Tool

A Python tool for predicting the optimal temperature (Topt) of enzymes using
multiple prediction models and a machine learning-based consensus approach.

This package provides functionality to:
- Integrate multiple enzyme temperature prediction models
- Generate consensus predictions using machine learning
- Provide a user-friendly command-line interface
- Support extensible architecture for new models
"""

__version__ = "1.0.0"
__author__ = "EMSL Summer School"
__email__ = "contact@example.com"

from .predictors.base_predictor import BasePredictor, PredictionResult
from .consensus.consensus_model import ConsensusModel
from .utils.data_loader import DataLoader, SequenceValidator

__all__ = [
    "BasePredictor",
    "PredictionResult", 
    "ConsensusModel",
    "DataLoader",
    "SequenceValidator"
]
