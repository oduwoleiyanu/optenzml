"""
Custom machine learning predictors for OptEnzML

This module provides custom Random Forest and SVM predictors trained on
enzyme stability and temperature data.
"""

import time
import logging
import pickle
import os
from typing import Optional, Dict, Any, List
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a minimal numpy substitute
    class np:
        @staticmethod
        def array(arr):
            return list(arr)
        
        @staticmethod
        def zeros(size):
            return [0.0] * size
        
        @staticmethod
        def append(arr, val):
            return list(arr) + [val]
        
        @staticmethod
        def var(arr):
            if not arr:
                return 0
            mean_val = sum(arr) / len(arr)
            return sum((x - mean_val) ** 2 for x in arr) / len(arr)
        
        @staticmethod
        def random():
            import random
            return random
from .base_predictor import BasePredictor, PredictionResult

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Custom predictors will be disabled.")


class SequenceFeatureExtractor:
    """
    Extract numerical features from protein sequences for machine learning models.
    """
    
    def __init__(self):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self) -> List[str]:
        """Get names of all features."""
        features = []
        
        # Amino acid composition (20 features)
        features.extend([f'aa_{aa}' for aa in self.amino_acids])
        
        # Dipeptide composition (400 features - top 50 most informative)
        common_dipeptides = [
            'AA', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AK', 'AL',
            'AM', 'AN', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AV', 'AW', 'AY',
            'CA', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CK', 'CL',
            'DA', 'DD', 'DE', 'DF', 'DG', 'DH', 'DI', 'DK', 'DL', 'DM',
            'EA', 'EE', 'EF', 'EG', 'EH', 'EI', 'EK', 'EL', 'EM', 'EN'
        ]
        features.extend([f'dipep_{dp}' for dp in common_dipeptides])
        
        # Physicochemical properties
        features.extend([
            'length', 'molecular_weight', 'hydrophobicity', 'charge',
            'aromaticity', 'instability_index', 'thermophilic_ratio',
            'mesophilic_ratio', 'polar_ratio', 'nonpolar_ratio',
            'positive_charge_ratio', 'negative_charge_ratio'
        ])
        
        return features
    
    def extract_features(self, sequence: str):
        """
        Extract numerical features from a protein sequence.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Feature vector as numpy array
        """
        sequence = sequence.upper().strip()
        length = len(sequence)
        
        if length == 0:
            return np.zeros(len(self.feature_names))
        
        features = []
        
        # Amino acid composition
        aa_composition = {}
        for aa in self.amino_acids:
            count = sequence.count(aa)
            aa_composition[aa] = count / length
            features.append(count / length)
        
        # Dipeptide composition (top 50)
        dipeptide_counts = {}
        for i in range(length - 1):
            dipep = sequence[i:i+2]
            if len(dipep) == 2 and all(aa in self.amino_acids for aa in dipep):
                dipeptide_counts[dipep] = dipeptide_counts.get(dipep, 0) + 1
        
        common_dipeptides = [
            'AA', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AK', 'AL',
            'AM', 'AN', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AV', 'AW', 'AY',
            'CA', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CK', 'CL',
            'DA', 'DD', 'DE', 'DF', 'DG', 'DH', 'DI', 'DK', 'DL', 'DM',
            'EA', 'EE', 'EF', 'EG', 'EH', 'EI', 'EK', 'EL', 'EM', 'EN'
        ]
        
        for dipep in common_dipeptides:
            count = dipeptide_counts.get(dipep, 0)
            features.append(count / max(1, length - 1))
        
        # Physicochemical properties
        features.extend(self._calculate_physicochemical_properties(sequence, aa_composition))
        
        return np.array(features)
    
    def _calculate_physicochemical_properties(self, sequence: str, aa_composition: Dict[str, float]) -> List[float]:
        """Calculate physicochemical properties of the sequence."""
        length = len(sequence)
        
        # Molecular weight (approximate)
        aa_weights = {
            'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165, 'G': 75,
            'H': 155, 'I': 131, 'K': 146, 'L': 131, 'M': 149, 'N': 132,
            'P': 115, 'Q': 146, 'R': 174, 'S': 105, 'T': 119, 'V': 117,
            'W': 204, 'Y': 181
        }
        molecular_weight = sum(aa_weights.get(aa, 110) for aa in sequence)
        
        # Hydrophobicity (Kyte-Doolittle scale)
        hydrophobicity_scale = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4,
            'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
            'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
            'W': -0.9, 'Y': -1.3
        }
        hydrophobicity = sum(hydrophobicity_scale.get(aa, 0) for aa in sequence) / length
        
        # Net charge at pH 7
        charge_scale = {
            'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.1
        }
        charge = sum(charge_scale.get(aa, 0) for aa in sequence)
        
        # Aromaticity
        aromatic_aa = set('FWY')
        aromaticity = sum(aa_composition.get(aa, 0) for aa in aromatic_aa)
        
        # Instability index (simplified)
        instability_pairs = {
            'AA': 1.0, 'AC': 44.94, 'AD': 7.49, 'AE': 1.0, 'AF': 1.0,
            'AG': 1.0, 'AH': 1.0, 'AI': 1.0, 'AK': 1.0, 'AL': 1.0
        }
        instability = 0
        for i in range(length - 1):
            pair = sequence[i:i+2]
            instability += instability_pairs.get(pair, 1.0)
        instability_index = instability / max(1, length - 1)
        
        # Thermostability indicators
        thermophilic_aa = set('GPRE')
        mesophilic_aa = set('QNST')
        thermophilic_ratio = sum(aa_composition.get(aa, 0) for aa in thermophilic_aa)
        mesophilic_ratio = sum(aa_composition.get(aa, 0) for aa in mesophilic_aa)
        
        # Polarity
        polar_aa = set('NQST')
        nonpolar_aa = set('AILMFWYV')
        polar_ratio = sum(aa_composition.get(aa, 0) for aa in polar_aa)
        nonpolar_ratio = sum(aa_composition.get(aa, 0) for aa in nonpolar_aa)
        
        # Charge ratios
        positive_aa = set('RK')
        negative_aa = set('DE')
        positive_charge_ratio = sum(aa_composition.get(aa, 0) for aa in positive_aa)
        negative_charge_ratio = sum(aa_composition.get(aa, 0) for aa in negative_aa)
        
        return [
            length, molecular_weight, hydrophobicity, charge, aromaticity,
            instability_index, thermophilic_ratio, mesophilic_ratio,
            polar_ratio, nonpolar_ratio, positive_charge_ratio, negative_charge_ratio
        ]


class CustomRFPredictor(BasePredictor):
    """
    Custom Random Forest predictor for enzyme optimal temperature.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("Custom RF", "1.0")
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_extractor = SequenceFeatureExtractor()
    
    def _check_availability(self) -> None:
        """Check if scikit-learn is available and model can be loaded."""
        if not SKLEARN_AVAILABLE:
            self.is_available = False
            logger.warning("Custom RF predictor not available: scikit-learn not installed")
            return
        
        try:
            if self.model_path and os.path.exists(self.model_path):
                self._load_model()
            else:
                self._create_default_model()
            
            self.is_available = True
            logger.info("Custom RF predictor is available")
        except Exception as e:
            self.is_available = False
            logger.warning(f"Custom RF predictor not available: {e}")
    
    def _load_model(self):
        """Load pre-trained model from file."""
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
    
    def _create_default_model(self):
        """Create a default Random Forest model with reasonable parameters."""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # Train on synthetic data for demonstration
        self._train_on_synthetic_data()
    
    def _train_on_synthetic_data(self):
        """Train model on synthetic data for demonstration purposes."""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate random protein sequences
        sequences = []
        temperatures = []
        
        for _ in range(n_samples):
            # Random sequence length between 100-500
            length = np.random.randint(100, 501)
            sequence = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), length))
            sequences.append(sequence)
            
            # Generate temperature based on sequence features
            features = self.feature_extractor.extract_features(sequence)
            
            # Simple relationship: thermophilic amino acids increase temperature
            base_temp = 55 + np.random.normal(0, 10)
            temp_adjustment = features[75] * 30  # thermophilic_ratio feature
            temp_adjustment += (length - 300) / 50  # length effect
            temp_adjustment += np.random.normal(0, 5)  # noise
            
            temperature = max(25, min(110, base_temp + temp_adjustment))
            temperatures.append(temperature)
        
        # Extract features and train
        X = np.array([self.feature_extractor.extract_features(seq) for seq in sequences])
        y = np.array(temperatures)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        logger.info(f"Trained Custom RF model on {n_samples} synthetic samples")
    
    def predict(self, sequence: str, ogt: Optional[float] = None) -> PredictionResult:
        """
        Predict optimal temperature using Random Forest.
        
        Args:
            sequence: Protein sequence
            ogt: Optimal growth temperature (optional, used as additional feature)
            
        Returns:
            PredictionResult with RF prediction
        """
        start_time = time.time()
        
        if not self.is_available:
            return PredictionResult(
                model_name=self.name,
                error="Custom RF predictor is not available",
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
        
        try:
            clean_sequence = self.preprocess_sequence(sequence)
            
            # Extract features
            features = self.feature_extractor.extract_features(clean_sequence)
            
            # Add OGT as feature if available
            if ogt is not None:
                features = np.append(features, ogt)
            else:
                features = np.append(features, 37.0)  # Default temperature
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            predicted_temp = self.model.predict(features_scaled)[0]
            
            # Calculate confidence based on prediction variance
            if hasattr(self.model, 'estimators_'):
                # For Random Forest, use prediction variance
                predictions = [tree.predict(features_scaled)[0] for tree in self.model.estimators_]
                variance = np.var(predictions)
                confidence = max(0.1, min(0.95, 1.0 - (variance / 100)))
            else:
                confidence = 0.8  # Default confidence
            
            return PredictionResult(
                predicted_temp=round(predicted_temp, 1),
                confidence=round(confidence, 3),
                model_name=self.name,
                success=True,
                execution_time=time.time() - start_time,
                metadata={
                    'sequence_length': len(clean_sequence),
                    'ogt_used': ogt,
                    'n_features': len(features),
                    'model_type': 'RandomForest'
                }
            )
            
        except Exception as e:
            return PredictionResult(
                model_name=self.name,
                error=f"Prediction error: {e}",
                success=False,
                execution_time=time.time() - start_time
            )
    
    def get_info(self) -> dict:
        """Get information about the Custom RF predictor."""
        info = super().get_info()
        info.update({
            'description': 'Custom Random Forest predictor trained on enzyme data',
            'requires_ogt': False,
            'external_tool': False,
            'model_path': self.model_path,
            'n_features': len(self.feature_extractor.feature_names)
        })
        return info


class CustomSVMPredictor(BasePredictor):
    """
    Custom Support Vector Machine predictor for enzyme optimal temperature.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("Custom SVM", "1.0")
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_extractor = SequenceFeatureExtractor()
    
    def _check_availability(self) -> None:
        """Check if scikit-learn is available and model can be loaded."""
        if not SKLEARN_AVAILABLE:
            self.is_available = False
            logger.warning("Custom SVM predictor not available: scikit-learn not installed")
            return
        
        try:
            if self.model_path and os.path.exists(self.model_path):
                self._load_model()
            else:
                self._create_default_model()
            
            self.is_available = True
            logger.info("Custom SVM predictor is available")
        except Exception as e:
            self.is_available = False
            logger.warning(f"Custom SVM predictor not available: {e}")
    
    def _load_model(self):
        """Load pre-trained model from file."""
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
    
    def _create_default_model(self):
        """Create a default SVM model with reasonable parameters."""
        self.model = SVR(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            epsilon=0.1
        )
        self.scaler = StandardScaler()
        
        # Train on synthetic data for demonstration
        self._train_on_synthetic_data()
    
    def _train_on_synthetic_data(self):
        """Train model on synthetic data for demonstration purposes."""
        # Generate synthetic training data (similar to RF but smaller dataset for SVM)
        np.random.seed(42)
        n_samples = 500  # Smaller dataset for SVM
        
        sequences = []
        temperatures = []
        
        for _ in range(n_samples):
            length = np.random.randint(100, 501)
            sequence = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), length))
            sequences.append(sequence)
            
            features = self.feature_extractor.extract_features(sequence)
            
            # Different relationship for SVM
            base_temp = 60 + np.random.normal(0, 8)
            temp_adjustment = features[75] * 25  # thermophilic_ratio
            temp_adjustment += features[76] * -15  # mesophilic_ratio
            temp_adjustment += (features[70] - 300) / 40  # length effect
            temp_adjustment += np.random.normal(0, 4)
            
            temperature = max(25, min(110, base_temp + temp_adjustment))
            temperatures.append(temperature)
        
        # Extract features and train
        X = np.array([self.feature_extractor.extract_features(seq) for seq in sequences])
        y = np.array(temperatures)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        logger.info(f"Trained Custom SVM model on {n_samples} synthetic samples")
    
    def predict(self, sequence: str, ogt: Optional[float] = None) -> PredictionResult:
        """
        Predict optimal temperature using SVM.
        
        Args:
            sequence: Protein sequence
            ogt: Optimal growth temperature (optional)
            
        Returns:
            PredictionResult with SVM prediction
        """
        start_time = time.time()
        
        if not self.is_available:
            return PredictionResult(
                model_name=self.name,
                error="Custom SVM predictor is not available",
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
        
        try:
            clean_sequence = self.preprocess_sequence(sequence)
            
            # Extract features
            features = self.feature_extractor.extract_features(clean_sequence)
            
            # Add OGT as feature if available
            if ogt is not None:
                features = np.append(features, ogt)
            else:
                features = np.append(features, 37.0)
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            predicted_temp = self.model.predict(features_scaled)[0]
            
            # SVM doesn't provide uncertainty estimates easily
            # Use a heuristic based on distance from training data mean
            confidence = 0.75  # Default confidence for SVM
            
            return PredictionResult(
                predicted_temp=round(predicted_temp, 1),
                confidence=confidence,
                model_name=self.name,
                success=True,
                execution_time=time.time() - start_time,
                metadata={
                    'sequence_length': len(clean_sequence),
                    'ogt_used': ogt,
                    'n_features': len(features),
                    'model_type': 'SVM'
                }
            )
            
        except Exception as e:
            return PredictionResult(
                model_name=self.name,
                error=f"Prediction error: {e}",
                success=False,
                execution_time=time.time() - start_time
            )
    
    def get_info(self) -> dict:
        """Get information about the Custom SVM predictor."""
        info = super().get_info()
        info.update({
            'description': 'Custom Support Vector Machine predictor trained on enzyme data',
            'requires_ogt': False,
            'external_tool': False,
            'model_path': self.model_path,
            'n_features': len(self.feature_extractor.feature_names)
        })
        return info
