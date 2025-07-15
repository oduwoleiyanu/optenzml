"""
Predictors module for OptEnzML

This module contains all the individual prediction models and the base predictor class.
"""

from .base_predictor import BasePredictor, PredictionResult
from .tomer_predictor import TomerPredictor
from .seq2topt_predictor import Seq2ToptPredictor
# Import custom predictors conditionally
try:
    from .custom_predictor import CustomRFPredictor, CustomSVMPredictor
    CUSTOM_PREDICTORS_AVAILABLE = True
except ImportError:
    CUSTOM_PREDICTORS_AVAILABLE = False
    # Create dummy classes
    class CustomRFPredictor:
        def __init__(self, *args, **kwargs):
            pass
    class CustomSVMPredictor:
        def __init__(self, *args, **kwargs):
            pass

__all__ = [
    "BasePredictor",
    "PredictionResult",
    "TomerPredictor", 
    "Seq2ToptPredictor",
    "CustomRFPredictor",
    "CustomSVMPredictor"
]
