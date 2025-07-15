"""
TOMER predictor wrapper for OptEnzML

This module provides a wrapper for the TOMER (Temperature Optimum Enzyme Regression) tool,
allowing it to be integrated into the OptEnzML consensus system.
"""

import subprocess
import tempfile
import os
import time
import logging
from typing import Optional
from .base_predictor import BasePredictor, PredictionResult

logger = logging.getLogger(__name__)


class TomerPredictor(BasePredictor):
    """
    Wrapper for the TOMER enzyme temperature prediction tool.
    
    TOMER is an external tool that predicts enzyme optimal temperature
    based on protein sequence and optimal growth temperature (OGT).
    """
    
    def __init__(self, tomer_path: Optional[str] = None):
        """
        Initialize the TOMER predictor.
        
        Args:
            tomer_path: Path to the TOMER executable (optional)
        """
        super().__init__("TOMER", "1.0")
        self.tomer_path = tomer_path or "tomer"  # Assume it's in PATH if not specified
        self.temp_dir = tempfile.gettempdir()
    
    def _check_availability(self) -> None:
        """
        Check if TOMER is available and executable.
        """
        try:
            # Try to run TOMER with help flag to check if it's available
            result = subprocess.run(
                [self.tomer_path, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            self.is_available = result.returncode == 0
            if self.is_available:
                logger.info("TOMER predictor is available")
            else:
                logger.warning(f"TOMER predictor not available: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            self.is_available = False
            logger.warning(f"TOMER predictor not available: {e}")
    
    def predict(self, sequence: str, ogt: Optional[float] = None) -> PredictionResult:
        """
        Predict optimal temperature using TOMER.
        
        Args:
            sequence: Protein sequence
            ogt: Optimal growth temperature of source organism
            
        Returns:
            PredictionResult with TOMER prediction
        """
        start_time = time.time()
        
        # Validate inputs
        if not self.is_available:
            return PredictionResult(
                model_name=self.name,
                error="TOMER predictor is not available",
                success=False,
                execution_time=time.time() - start_time
            )
        
        if not self.validate_sequence(sequence):
            return PredictionResult(
                model_name=self.name,
                error="Invalid protein sequence",
                success=False,
                execution_time=time.time() - start_time
            )
        
        if ogt is None:
            return PredictionResult(
                model_name=self.name,
                error="TOMER requires optimal growth temperature (OGT)",
                success=False,
                execution_time=time.time() - start_time
            )
        
        # Preprocess sequence
        clean_sequence = self.preprocess_sequence(sequence)
        
        try:
            # Create temporary files for input and output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as temp_input:
                temp_input.write(f">sequence\n{clean_sequence}\n")
                temp_input_path = temp_input.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_output:
                temp_output_path = temp_output.name
            
            try:
                # Run TOMER
                cmd = [
                    self.tomer_path,
                    "--input", temp_input_path,
                    "--output", temp_output_path,
                    "--ogt", str(ogt)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60  # 1 minute timeout
                )
                
                if result.returncode != 0:
                    return PredictionResult(
                        model_name=self.name,
                        error=f"TOMER execution failed: {result.stderr}",
                        success=False,
                        execution_time=time.time() - start_time
                    )
                
                # Parse output
                predicted_temp = self._parse_tomer_output(temp_output_path)
                
                if predicted_temp is None:
                    return PredictionResult(
                        model_name=self.name,
                        error="Failed to parse TOMER output",
                        success=False,
                        execution_time=time.time() - start_time
                    )
                
                return PredictionResult(
                    predicted_temp=predicted_temp,
                    model_name=self.name,
                    success=True,
                    execution_time=time.time() - start_time,
                    metadata={
                        'ogt_used': ogt,
                        'sequence_length': len(clean_sequence),
                        'tomer_version': self.version
                    }
                )
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(temp_input_path)
                    os.unlink(temp_output_path)
                except OSError:
                    pass
                    
        except subprocess.TimeoutExpired:
            return PredictionResult(
                model_name=self.name,
                error="TOMER execution timed out",
                success=False,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return PredictionResult(
                model_name=self.name,
                error=f"Unexpected error running TOMER: {e}",
                success=False,
                execution_time=time.time() - start_time
            )
    
    def _parse_tomer_output(self, output_path: str) -> Optional[float]:
        """
        Parse TOMER output file to extract predicted temperature.
        
        Args:
            output_path: Path to TOMER output file
            
        Returns:
            Predicted temperature or None if parsing failed
        """
        try:
            with open(output_path, 'r') as f:
                content = f.read().strip()
            
            # TOMER output format may vary - this is a generic parser
            # Look for temperature values in the output
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Try to extract numeric value
                    parts = line.split()
                    for part in parts:
                        try:
                            temp = float(part)
                            # Reasonable temperature range for enzymes
                            if 0 <= temp <= 150:
                                return temp
                        except ValueError:
                            continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing TOMER output: {e}")
            return None
    
    def get_info(self) -> dict:
        """Get information about the TOMER predictor."""
        info = super().get_info()
        info.update({
            'description': 'TOMER: Thermostability predictor for enzymes',
            'requires_ogt': True,
            'external_tool': True,
            'tomer_path': getattr(self, 'tomer_path', None)
        })
        return info


class MockTomerPredictor(BasePredictor):
    """
    Mock TOMER predictor for testing and demonstration purposes.
    
    This predictor simulates TOMER behavior when the actual tool is not available.
    It uses a simple heuristic based on sequence composition and OGT.
    """
    
    def __init__(self):
        super().__init__("TOMER (Mock)", "1.0")
    
    def _check_availability(self) -> None:
        """Mock TOMER is always available."""
        self.is_available = True
        logger.info("Mock TOMER predictor is available")
    
    def predict(self, sequence: str, ogt: Optional[float] = None) -> PredictionResult:
        """
        Mock prediction based on simple heuristics.
        
        Args:
            sequence: Protein sequence
            ogt: Optimal growth temperature
            
        Returns:
            PredictionResult with mock prediction
        """
        start_time = time.time()
        
        if not self.validate_sequence(sequence):
            return PredictionResult(
                model_name=self.name,
                error="Invalid protein sequence",
                success=False,
                execution_time=time.time() - start_time
            )
        
        if ogt is None:
            ogt = 37.0  # Default human body temperature
        
        clean_sequence = self.preprocess_sequence(sequence)
        
        # Simple heuristic: base prediction on OGT and sequence composition
        # Thermophilic amino acids: G, P, R, E (higher in thermophiles)
        thermophilic_aa = set('GPRE')
        thermophilic_count = sum(1 for aa in clean_sequence if aa in thermophilic_aa)
        thermophilic_ratio = thermophilic_count / len(clean_sequence)
        
        # Base prediction on OGT with adjustment for sequence composition
        predicted_temp = ogt + (thermophilic_ratio * 20) + (len(clean_sequence) / 1000)
        
        # Add some realistic noise
        import random
        predicted_temp += random.uniform(-5, 5)
        
        # Ensure reasonable range
        predicted_temp = max(20, min(120, predicted_temp))
        
        return PredictionResult(
            predicted_temp=round(predicted_temp, 1),
            confidence=0.7,  # Mock confidence
            model_name=self.name,
            success=True,
            execution_time=time.time() - start_time,
            metadata={
                'ogt_used': ogt,
                'sequence_length': len(clean_sequence),
                'thermophilic_ratio': round(thermophilic_ratio, 3),
                'mock_prediction': True
            }
        )
    
    def get_info(self) -> dict:
        """Get information about the mock TOMER predictor."""
        info = super().get_info()
        info.update({
            'description': 'Mock TOMER predictor for testing and demonstration',
            'requires_ogt': True,
            'external_tool': False,
            'mock': True
        })
        return info
