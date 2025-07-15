"""
Consensus model for OptEnzML

This module implements the consensus prediction system that combines outputs
from multiple individual predictors using machine learning meta-models.
"""

import time
import logging
import pickle
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a minimal numpy substitute for basic operations
    class np:
        @staticmethod
        def mean(arr):
            if not arr:
                return 0.0
            if isinstance(arr, list):
                return sum(arr) / len(arr)
            else:
                # Handle case where arr might be a nested structure
                flat_arr = []
                def flatten(x):
                    if isinstance(x, (list, tuple)):
                        for item in x:
                            flatten(item)
                    else:
                        flat_arr.append(x)
                flatten(arr)
                return sum(flat_arr) / len(flat_arr) if flat_arr else 0.0
        
        @staticmethod
        def var(arr):
            if not arr:
                return 0
            mean_val = sum(arr) / len(arr)
            return sum((x - mean_val) ** 2 for x in arr) / len(arr)
        
        @staticmethod
        def std(arr):
            return np.var(arr) ** 0.5
        
        @staticmethod
        def min(arr):
            return min(arr) if arr else 0
        
        @staticmethod
        def max(arr):
            return max(arr) if arr else 0
        
        @staticmethod
        def sum(arr):
            return sum(arr) if arr else 0
        
        @staticmethod
        def average(arr, weights=None):
            if not arr:
                return 0.0
            if weights is None:
                return sum(arr) / len(arr)
            else:
                if len(arr) != len(weights):
                    return sum(arr) / len(arr)
                weighted_sum = sum(a * w for a, w in zip(arr, weights))
                weight_sum = sum(weights)
                return weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        @staticmethod
        def array(arr):
            if isinstance(arr, (list, tuple)):
                return list(arr)
            else:
                return [arr]
        
        @staticmethod
        def ones(size):
            if isinstance(size, int):
                return [1.0] * size
            else:
                return [1.0]
from ..predictors.base_predictor import BasePredictor, PredictionResult

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Consensus model will use simple averaging.")


@dataclass
class ConsensusResult:
    """
    Data class to store consensus prediction results.
    
    Attributes:
        consensus_temp: Final consensus temperature prediction
        confidence: Confidence score for the consensus prediction
        individual_predictions: List of individual predictor results
        consensus_method: Method used for consensus (e.g., 'weighted_average', 'ml_ensemble')
        execution_time: Total time for consensus prediction
        metadata: Additional information about the consensus process
        success: Whether the consensus prediction was successful
        error: Error message if consensus failed
    """
    consensus_temp: Optional[float] = None
    confidence: Optional[float] = None
    individual_predictions: List[PredictionResult] = None
    consensus_method: str = ""
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = None
    success: bool = False
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.individual_predictions is None:
            self.individual_predictions = []
        if self.metadata is None:
            self.metadata = {}


class ConsensusModel:
    """
    Consensus model that combines predictions from multiple individual predictors.
    
    This class implements various consensus strategies including simple averaging,
    weighted averaging, and machine learning-based meta-models.
    """
    
    def __init__(self, predictors: List[BasePredictor], consensus_model_path: Optional[str] = None):
        """
        Initialize the consensus model.
        
        Args:
            predictors: List of individual predictor instances
            consensus_model_path: Path to pre-trained consensus model (optional)
        """
        self.predictors = predictors
        self.consensus_model_path = consensus_model_path
        self.meta_model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        
        # Filter available predictors
        self.available_predictors = [p for p in self.predictors if p.is_available]
        
        if not self.available_predictors:
            logger.warning("No available predictors found for consensus model")
        else:
            logger.info(f"Consensus model initialized with {len(self.available_predictors)} available predictors")
        
        # Load pre-trained model if available
        if self.consensus_model_path and os.path.exists(self.consensus_model_path):
            self._load_consensus_model()
    
    def _load_consensus_model(self):
        """Load pre-trained consensus model from file."""
        try:
            with open(self.consensus_model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.meta_model = model_data['model']
                self.scaler = model_data.get('scaler')
                self.feature_names = model_data.get('feature_names', [])
                self.is_trained = True
            logger.info(f"Loaded pre-trained consensus model from {self.consensus_model_path}")
        except Exception as e:
            logger.warning(f"Failed to load consensus model: {e}")
            self.is_trained = False
    
    def predict(self, sequence: str, ogt: Optional[float] = None, 
                method: str = 'auto') -> ConsensusResult:
        """
        Generate consensus prediction from multiple predictors.
        
        Args:
            sequence: Protein sequence
            ogt: Optimal growth temperature (optional)
            method: Consensus method ('auto', 'simple_average', 'weighted_average', 'ml_ensemble')
            
        Returns:
            ConsensusResult with consensus prediction and individual results
        """
        start_time = time.time()
        
        if not self.available_predictors:
            return ConsensusResult(
                error="No available predictors for consensus",
                success=False,
                execution_time=time.time() - start_time
            )
        
        # Get predictions from all available predictors
        individual_predictions = []
        for predictor in self.available_predictors:
            try:
                result = predictor.predict(sequence, ogt)
                individual_predictions.append(result)
            except Exception as e:
                logger.warning(f"Error getting prediction from {predictor.name}: {e}")
                # Add failed prediction result
                failed_result = PredictionResult(
                    model_name=predictor.name,
                    error=str(e),
                    success=False
                )
                individual_predictions.append(failed_result)
        
        # Filter successful predictions
        successful_predictions = [p for p in individual_predictions if p.success and p.predicted_temp is not None]
        
        if not successful_predictions:
            return ConsensusResult(
                individual_predictions=individual_predictions,
                error="No successful individual predictions",
                success=False,
                execution_time=time.time() - start_time
            )
        
        # Choose consensus method
        if method == 'auto':
            if self.is_trained and SKLEARN_AVAILABLE:
                method = 'ml_ensemble'
            elif len(successful_predictions) >= 2:
                method = 'weighted_average'
            else:
                method = 'simple_average'
        
        # Generate consensus prediction
        try:
            if method == 'ml_ensemble' and self.is_trained:
                consensus_temp, confidence = self._ml_ensemble_prediction(successful_predictions)
            elif method == 'weighted_average':
                consensus_temp, confidence = self._weighted_average_prediction(successful_predictions)
            else:
                consensus_temp, confidence = self._simple_average_prediction(successful_predictions)
            
            return ConsensusResult(
                consensus_temp=round(consensus_temp, 1),
                confidence=round(confidence, 3),
                individual_predictions=individual_predictions,
                consensus_method=method,
                execution_time=time.time() - start_time,
                metadata={
                    'n_total_predictors': len(self.available_predictors),
                    'n_successful_predictions': len(successful_predictions),
                    'prediction_variance': np.var([p.predicted_temp for p in successful_predictions]),
                    'sequence_length': len(sequence)
                },
                success=True
            )
            
        except Exception as e:
            return ConsensusResult(
                individual_predictions=individual_predictions,
                error=f"Consensus prediction failed: {e}",
                success=False,
                execution_time=time.time() - start_time
            )
    
    def _simple_average_prediction(self, predictions: List[PredictionResult]) -> Tuple[float, float]:
        """Simple average of all predictions."""
        temps = [p.predicted_temp for p in predictions]
        consensus_temp = np.mean(temps)
        
        # Confidence based on agreement between predictors
        variance = np.var(temps)
        confidence = max(0.1, min(0.9, 1.0 - (variance / 100)))
        
        return consensus_temp, confidence
    
    def _weighted_average_prediction(self, predictions: List[PredictionResult]) -> Tuple[float, float]:
        """Weighted average based on individual predictor confidence scores."""
        temps = []
        weights = []
        
        for pred in predictions:
            temps.append(pred.predicted_temp)
            # Use confidence if available, otherwise use default weight
            weight = pred.confidence if pred.confidence is not None else 0.5
            weights.append(weight)
        
        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            normalized_weights = [w / weight_sum for w in weights]
        else:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        
        # Weighted average
        consensus_temp = sum(t * w for t, w in zip(temps, normalized_weights))
        
        # Confidence based on simple variance and average confidence
        mean_temp = sum(temps) / len(temps)
        variance = sum((t - mean_temp) ** 2 for t in temps) / len(temps)
        avg_confidence = sum(p.confidence or 0.5 for p in predictions) / len(predictions)
        
        # Combine variance-based and average confidence
        variance_confidence = max(0.1, min(0.9, 1.0 - (variance / 50)))
        confidence = (variance_confidence + avg_confidence) / 2
        
        return consensus_temp, confidence
    
    def _ml_ensemble_prediction(self, predictions: List[PredictionResult]) -> Tuple[float, float]:
        """Machine learning ensemble prediction using trained meta-model."""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            # Fallback to weighted average
            return self._weighted_average_prediction(predictions)
        
        # Extract features for meta-model
        features = self._extract_meta_features(predictions)
        
        if len(features) != len(self.feature_names):
            logger.warning("Feature dimension mismatch, falling back to weighted average")
            return self._weighted_average_prediction(predictions)
        
        # Scale features if scaler is available
        if self.scaler:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        # Make prediction
        consensus_temp = self.meta_model.predict(features_scaled)[0]
        
        # Estimate confidence (simplified)
        if hasattr(self.meta_model, 'estimators_'):
            # For Random Forest, use prediction variance
            tree_predictions = [tree.predict(features_scaled)[0] for tree in self.meta_model.estimators_]
            variance = np.var(tree_predictions)
            confidence = max(0.2, min(0.95, 1.0 - (variance / 50)))
        else:
            # Default confidence for other models
            confidence = 0.8
        
        return consensus_temp, confidence
    
    def _extract_meta_features(self, predictions: List[PredictionResult]):
        """Extract features for meta-model from individual predictions."""
        features = []
        
        # Individual predictions
        for pred in predictions:
            features.append(pred.predicted_temp if pred.predicted_temp is not None else 0.0)
        
        # Pad or truncate to expected number of predictors
        expected_predictors = len(self.predictors)
        while len(features) < expected_predictors:
            features.append(0.0)  # Padding for missing predictors
        features = features[:expected_predictors]
        
        # Individual confidences
        for pred in predictions:
            features.append(pred.confidence if pred.confidence is not None else 0.5)
        
        # Pad confidences
        while len(features) < 2 * expected_predictors:
            features.append(0.5)
        features = features[:2 * expected_predictors]
        
        # Statistical features
        valid_temps = [p.predicted_temp for p in predictions if p.predicted_temp is not None]
        if valid_temps:
            features.extend([
                np.mean(valid_temps),
                np.std(valid_temps),
                np.min(valid_temps),
                np.max(valid_temps),
                len(valid_temps)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def get_predictor_info(self) -> List[Dict[str, Any]]:
        """Get information about all predictors."""
        return [predictor.get_info() for predictor in self.predictors]
    
    def get_available_predictors(self) -> List[str]:
        """Get names of available predictors."""
        return [predictor.name for predictor in self.available_predictors]
    
    def get_consensus_info(self) -> Dict[str, Any]:
        """Get information about the consensus model."""
        return {
            'total_predictors': len(self.predictors),
            'available_predictors': len(self.available_predictors),
            'predictor_names': [p.name for p in self.available_predictors],
            'is_trained': self.is_trained,
            'consensus_model_path': self.consensus_model_path,
            'sklearn_available': SKLEARN_AVAILABLE
        }
    
    def batch_predict(self, sequences: List[str], ogts: Optional[List[float]] = None,
                     method: str = 'auto') -> List[ConsensusResult]:
        """
        Generate consensus predictions for multiple sequences.
        
        Args:
            sequences: List of protein sequences
            ogts: List of optimal growth temperatures (optional)
            method: Consensus method to use
            
        Returns:
            List of ConsensusResult objects
        """
        results = []
        
        if ogts is None:
            ogts = [None] * len(sequences)
        elif len(ogts) != len(sequences):
            logger.warning(f"Number of OGTs ({len(ogts)}) doesn't match number of sequences ({len(sequences)})")
            ogts = ogts + [None] * (len(sequences) - len(ogts))
        
        for i, (sequence, ogt) in enumerate(zip(sequences, ogts)):
            try:
                result = self.predict(sequence, ogt, method)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting sequence {i+1}: {e}")
                error_result = ConsensusResult(
                    error=str(e),
                    success=False
                )
                results.append(error_result)
        
        return results
