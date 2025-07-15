"""
Seq2Topt predictor wrapper for OptEnzML

This module provides a wrapper for the Seq2Topt model, allowing it to be
integrated into the OptEnzML consensus system.
"""

import subprocess
import tempfile
import os
import time
import logging
import json
from typing import Optional
from .base_predictor import BasePredictor, PredictionResult

logger = logging.getLogger(__name__)


class Seq2ToptPredictor(BasePredictor):
    """
    Wrapper for the Seq2Topt enzyme temperature prediction model.
    
    Seq2Topt is a deep learning model that predicts enzyme optimal temperature
    directly from protein sequence without requiring OGT.
    """
    
    def __init__(self, seq2topt_path: Optional[str] = None, model_path: Optional[str] = None):
        """
        Initialize the Seq2Topt predictor.
        
        Args:
            seq2topt_path: Path to the Seq2Topt executable/script (optional)
            model_path: Path to the trained Seq2Topt model (optional)
        """
        super().__init__("Seq2Topt", "1.0")
        self.seq2topt_path = seq2topt_path or "seq2topt"
        self.model_path = model_path
        self.temp_dir = tempfile.gettempdir()
    
    def _check_availability(self) -> None:
        """
        Check if Seq2Topt is available and executable.
        """
        try:
            # Try to run Seq2Topt with help flag to check if it's available
            result = subprocess.run(
                [self.seq2topt_path, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            self.is_available = result.returncode == 0
            if self.is_available:
                logger.info("Seq2Topt predictor is available")
            else:
                logger.warning(f"Seq2Topt predictor not available: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            self.is_available = False
            logger.warning(f"Seq2Topt predictor not available: {e}")
    
    def predict(self, sequence: str, ogt: Optional[float] = None) -> PredictionResult:
        """
        Predict optimal temperature using Seq2Topt.
        
        Args:
            sequence: Protein sequence
            ogt: Optimal growth temperature (not used by Seq2Topt but kept for interface consistency)
            
        Returns:
            PredictionResult with Seq2Topt prediction
        """
        start_time = time.time()
        
        # Validate inputs
        if not self.is_available:
            return PredictionResult(
                model_name=self.name,
                error="Seq2Topt predictor is not available",
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
        
        # Preprocess sequence
        clean_sequence = self.preprocess_sequence(sequence)
        
        try:
            # Create temporary files for input and output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as temp_input:
                temp_input.write(f">sequence\n{clean_sequence}\n")
                temp_input_path = temp_input.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_output:
                temp_output_path = temp_output.name
            
            try:
                # Build command
                cmd = [self.seq2topt_path, "--input", temp_input_path, "--output", temp_output_path]
                if self.model_path:
                    cmd.extend(["--model", self.model_path])
                
                # Run Seq2Topt
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout for deep learning model
                )
                
                if result.returncode != 0:
                    return PredictionResult(
                        model_name=self.name,
                        error=f"Seq2Topt execution failed: {result.stderr}",
                        success=False,
                        execution_time=time.time() - start_time
                    )
                
                # Parse output
                prediction_data = self._parse_seq2topt_output(temp_output_path)
                
                if prediction_data is None:
                    return PredictionResult(
                        model_name=self.name,
                        error="Failed to parse Seq2Topt output",
                        success=False,
                        execution_time=time.time() - start_time
                    )
                
                return PredictionResult(
                    predicted_temp=prediction_data['temperature'],
                    confidence=prediction_data.get('confidence'),
                    model_name=self.name,
                    success=True,
                    execution_time=time.time() - start_time,
                    metadata={
                        'sequence_length': len(clean_sequence),
                        'seq2topt_version': self.version,
                        'model_path': self.model_path,
                        **prediction_data.get('metadata', {})
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
                error="Seq2Topt execution timed out",
                success=False,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return PredictionResult(
                model_name=self.name,
                error=f"Unexpected error running Seq2Topt: {e}",
                success=False,
                execution_time=time.time() - start_time
            )
    
    def _parse_seq2topt_output(self, output_path: str) -> Optional[dict]:
        """
        Parse Seq2Topt output file to extract predicted temperature and metadata.
        
        Args:
            output_path: Path to Seq2Topt output file
            
        Returns:
            Dictionary with prediction data or None if parsing failed
        """
        try:
            with open(output_path, 'r') as f:
                content = f.read().strip()
            
            # Try to parse as JSON first
            try:
                data = json.loads(content)
                if 'temperature' in data or 'predicted_temperature' in data:
                    temp = data.get('temperature') or data.get('predicted_temperature')
                    return {
                        'temperature': float(temp),
                        'confidence': data.get('confidence'),
                        'metadata': data.get('metadata', {})
                    }
            except json.JSONDecodeError:
                pass
            
            # Fallback to text parsing
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Look for temperature values
                    parts = line.split()
                    for part in parts:
                        try:
                            temp = float(part)
                            # Reasonable temperature range for enzymes
                            if 0 <= temp <= 150:
                                return {'temperature': temp}
                        except ValueError:
                            continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing Seq2Topt output: {e}")
            return None
    
    def get_info(self) -> dict:
        """Get information about the Seq2Topt predictor."""
        info = super().get_info()
        info.update({
            'description': 'Seq2Topt deep learning model for enzyme temperature prediction',
            'requires_ogt': False,
            'external_tool': True,
            'seq2topt_path': self.seq2topt_path,
            'model_path': self.model_path
        })
        return info


class MockSeq2ToptPredictor(BasePredictor):
    """
    Mock Seq2Topt predictor for testing and demonstration purposes.
    
    This predictor simulates Seq2Topt behavior when the actual model is not available.
    It uses sequence-based features to make predictions.
    """
    
    def __init__(self):
        super().__init__("Seq2Topt (Mock)", "1.0")
    
    def _check_availability(self) -> None:
        """Mock Seq2Topt is always available."""
        self.is_available = True
        logger.info("Mock Seq2Topt predictor is available")
    
    def predict(self, sequence: str, ogt: Optional[float] = None) -> PredictionResult:
        """
        Mock prediction based on sequence features.
        
        Args:
            sequence: Protein sequence
            ogt: Optimal growth temperature (not used)
            
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
        
        clean_sequence = self.preprocess_sequence(sequence)
        
        # Calculate sequence features for mock prediction
        features = self._calculate_sequence_features(clean_sequence)
        
        # Mock deep learning prediction based on features
        predicted_temp = self._mock_deep_learning_prediction(features)
        
        # Mock confidence based on sequence length and composition
        confidence = min(0.95, 0.6 + (len(clean_sequence) / 1000) * 0.3)
        
        return PredictionResult(
            predicted_temp=round(predicted_temp, 1),
            confidence=round(confidence, 3),
            model_name=self.name,
            success=True,
            execution_time=time.time() - start_time,
            metadata={
                'sequence_length': len(clean_sequence),
                'features': features,
                'mock_prediction': True
            }
        )
    
    def _calculate_sequence_features(self, sequence: str) -> dict:
        """Calculate sequence features for mock prediction."""
        # Amino acid composition
        aa_counts = {}
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            aa_counts[aa] = sequence.count(aa) / len(sequence)
        
        # Thermostability indicators
        thermophilic_aa = set('GPRE')  # Glycine, Proline, Arginine, Glutamate
        mesophilic_aa = set('QNST')   # Glutamine, Asparagine, Serine, Threonine
        
        thermophilic_ratio = sum(aa_counts.get(aa, 0) for aa in thermophilic_aa)
        mesophilic_ratio = sum(aa_counts.get(aa, 0) for aa in mesophilic_aa)
        
        # Hydrophobicity
        hydrophobic_aa = set('AILMFWYV')
        hydrophobic_ratio = sum(aa_counts.get(aa, 0) for aa in hydrophobic_aa)
        
        # Charged residues
        positive_aa = set('RK')
        negative_aa = set('DE')
        charge_ratio = sum(aa_counts.get(aa, 0) for aa in positive_aa) - sum(aa_counts.get(aa, 0) for aa in negative_aa)
        
        return {
            'length': len(sequence),
            'thermophilic_ratio': round(thermophilic_ratio, 3),
            'mesophilic_ratio': round(mesophilic_ratio, 3),
            'hydrophobic_ratio': round(hydrophobic_ratio, 3),
            'charge_ratio': round(charge_ratio, 3),
            'gc_content': round((sequence.count('G') + sequence.count('C')) / len(sequence), 3)
        }
    
    def _mock_deep_learning_prediction(self, features: dict) -> float:
        """Mock deep learning prediction based on features."""
        # Base temperature
        base_temp = 55.0
        
        # Adjustments based on features
        temp_adjustment = 0
        
        # Thermophilic amino acids increase temperature
        temp_adjustment += features['thermophilic_ratio'] * 30
        
        # Mesophilic amino acids decrease temperature
        temp_adjustment -= features['mesophilic_ratio'] * 20
        
        # Hydrophobic residues slightly increase stability
        temp_adjustment += features['hydrophobic_ratio'] * 10
        
        # Charge interactions affect stability
        temp_adjustment += abs(features['charge_ratio']) * 15
        
        # Length effect (longer proteins tend to be more stable)
        if features['length'] > 300:
            temp_adjustment += 5
        elif features['length'] < 150:
            temp_adjustment -= 5
        
        predicted_temp = base_temp + temp_adjustment
        
        # Add some realistic noise
        import random
        predicted_temp += random.uniform(-3, 3)
        
        # Ensure reasonable range
        predicted_temp = max(25, min(110, predicted_temp))
        
        return predicted_temp
    
    def get_info(self) -> dict:
        """Get information about the mock Seq2Topt predictor."""
        info = super().get_info()
        info.update({
            'description': 'Mock Seq2Topt predictor for testing and demonstration',
            'requires_ogt': False,
            'external_tool': False,
            'mock': True
        })
        return info
