"""
Base predictor class for OptEnzML

This module defines the abstract base class that all prediction models must inherit from,
ensuring a consistent interface across different prediction methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """
    Data class to store prediction results from individual models.
    
    Attributes:
        predicted_temp: Predicted optimal temperature in Celsius
        confidence: Confidence score (0-1) if available
        model_name: Name of the prediction model
        execution_time: Time taken for prediction in seconds
        metadata: Additional model-specific information
        error: Error message if prediction failed
        success: Whether the prediction was successful
    """
    predicted_temp: Optional[float] = None
    confidence: Optional[float] = None
    model_name: str = ""
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    success: bool = False
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BasePredictor(ABC):
    """
    Abstract base class for all enzyme temperature predictors.
    
    This class defines the interface that all prediction models must implement,
    ensuring consistency and enabling the consensus system to work with any
    combination of predictors.
    """
    
    def __init__(self, name: str, version: str = "1.0"):
        """
        Initialize the predictor.
        
        Args:
            name: Human-readable name of the predictor
            version: Version of the predictor
        """
        self.name = name
        self.version = version
        self.is_available = False
        self._check_availability()
    
    @abstractmethod
    def _check_availability(self) -> None:
        """
        Check if the predictor is available and ready to use.
        
        This method should verify that all required dependencies, models,
        or external tools are available. Sets self.is_available accordingly.
        """
        pass
    
    @abstractmethod
    def predict(self, sequence: str, ogt: Optional[float] = None) -> PredictionResult:
        """
        Predict the optimal temperature for a given protein sequence.
        
        Args:
            sequence: Protein sequence as a string (single letter amino acid codes)
            ogt: Optimal growth temperature of the source organism (optional)
            
        Returns:
            PredictionResult object containing the prediction and metadata
        """
        pass
    
    def validate_sequence(self, sequence: str) -> bool:
        """
        Validate that the input sequence is a valid protein sequence.
        
        Args:
            sequence: Protein sequence to validate
            
        Returns:
            True if sequence is valid, False otherwise
        """
        if not sequence or not isinstance(sequence, str):
            return False
        
        # Remove whitespace and convert to uppercase
        sequence = sequence.strip().upper()
        
        # Check for valid amino acid characters
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        sequence_chars = set(sequence)
        
        # Allow some ambiguous amino acids
        valid_aa.update(['X', 'B', 'Z', 'J', 'U', 'O'])
        
        return sequence_chars.issubset(valid_aa) and len(sequence) > 0
    
    def preprocess_sequence(self, sequence: str) -> str:
        """
        Preprocess the protein sequence for prediction.
        
        Args:
            sequence: Raw protein sequence
            
        Returns:
            Cleaned and standardized protein sequence
        """
        if not sequence:
            return ""
        
        # Remove whitespace and convert to uppercase
        sequence = sequence.strip().upper()
        
        # Remove any non-amino acid characters except valid ambiguous codes
        valid_chars = set('ACDEFGHIKLMNPQRSTVWYXBZJUO')
        sequence = ''.join(char for char in sequence if char in valid_chars)
        
        return sequence
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the predictor.
        
        Returns:
            Dictionary containing predictor information
        """
        return {
            'name': self.name,
            'version': self.version,
            'available': self.is_available,
            'type': self.__class__.__name__
        }
    
    def batch_predict(self, sequences: List[str], ogts: Optional[List[float]] = None) -> List[PredictionResult]:
        """
        Predict optimal temperatures for multiple sequences.
        
        Args:
            sequences: List of protein sequences
            ogts: List of optimal growth temperatures (optional)
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        
        if ogts is None:
            ogts = [None] * len(sequences)
        elif len(ogts) != len(sequences):
            logger.warning(f"Number of OGTs ({len(ogts)}) doesn't match number of sequences ({len(sequences)})")
            ogts = ogts + [None] * (len(sequences) - len(ogts))
        
        for i, (sequence, ogt) in enumerate(zip(sequences, ogts)):
            try:
                result = self.predict(sequence, ogt)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting sequence {i+1}: {e}")
                error_result = PredictionResult(
                    model_name=self.name,
                    error=str(e),
                    success=False
                )
                results.append(error_result)
        
        return results
    
    def __str__(self) -> str:
        """String representation of the predictor."""
        status = "available" if self.is_available else "unavailable"
        return f"{self.name} v{self.version} ({status})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the predictor."""
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}', available={self.is_available})"
