#!/usr/bin/env python3
"""
Training script for custom OptEnzML predictors using BRENDA data.
This script trains Random Forest and SVM models on the curated BRENDA dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import joblib
from datetime import datetime

# Add the parent directory to the path to import optenzml modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Cannot train models.")

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class SequenceFeatureExtractor:
    """Extract numerical features from protein sequences for ML training."""
    
    def __init__(self):
        # Amino acid properties
        self.aa_properties = {
            'A': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'molecular_weight': 89.1},
            'R': {'hydrophobic': 0, 'polar': 1, 'charged': 1, 'aromatic': 0, 'molecular_weight': 174.2},
            'N': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 0, 'molecular_weight': 132.1},
            'D': {'hydrophobic': 0, 'polar': 1, 'charged': -1, 'aromatic': 0, 'molecular_weight': 133.1},
            'C': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'molecular_weight': 121.2},
            'Q': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 0, 'molecular_weight': 146.1},
            'E': {'hydrophobic': 0, 'polar': 1, 'charged': -1, 'aromatic': 0, 'molecular_weight': 147.1},
            'G': {'hydrophobic': 0, 'polar': 0, 'charged': 0, 'aromatic': 0, 'molecular_weight': 75.1},
            'H': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 1, 'molecular_weight': 155.2},
            'I': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'molecular_weight': 131.2},
            'L': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'molecular_weight': 131.2},
            'K': {'hydrophobic': 0, 'polar': 1, 'charged': 1, 'aromatic': 0, 'molecular_weight': 146.2},
            'M': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'molecular_weight': 149.2},
            'F': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 1, 'molecular_weight': 165.2},
            'P': {'hydrophobic': 0, 'polar': 0, 'charged': 0, 'aromatic': 0, 'molecular_weight': 115.1},
            'S': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 0, 'molecular_weight': 105.1},
            'T': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 0, 'molecular_weight': 119.1},
            'W': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 1, 'molecular_weight': 204.2},
            'Y': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 1, 'molecular_weight': 181.2},
            'V': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'molecular_weight': 117.1}
        }
    
    def extract_features(self, sequence: str) -> np.ndarray:
        """Extract numerical features from a protein sequence."""
        if not sequence:
            return np.zeros(20)  # Return zero vector for empty sequences
        
        # Basic sequence statistics
        length = len(sequence)
        
        # Amino acid composition
        aa_counts = {aa: sequence.count(aa) for aa in self.aa_properties.keys()}
        aa_frequencies = {aa: count/length for aa, count in aa_counts.items()}
        
        # Physicochemical properties
        hydrophobic_count = sum(1 for aa in sequence if self.aa_properties.get(aa, {}).get('hydrophobic', 0))
        polar_count = sum(1 for aa in sequence if self.aa_properties.get(aa, {}).get('polar', 0))
        charged_count = sum(1 for aa in sequence if abs(self.aa_properties.get(aa, {}).get('charged', 0)) > 0)
        aromatic_count = sum(1 for aa in sequence if self.aa_properties.get(aa, {}).get('aromatic', 0))
        
        # Molecular weight
        total_mw = sum(self.aa_properties.get(aa, {}).get('molecular_weight', 0) for aa in sequence)
        avg_mw = total_mw / length if length > 0 else 0
        
        # Compile features
        features = [
            length,
            hydrophobic_count / length,
            polar_count / length,
            charged_count / length,
            aromatic_count / length,
            avg_mw,
            # Top 10 most common amino acids frequencies
            aa_frequencies.get('A', 0), aa_frequencies.get('L', 0), aa_frequencies.get('G', 0),
            aa_frequencies.get('V', 0), aa_frequencies.get('E', 0), aa_frequencies.get('S', 0),
            aa_frequencies.get('I', 0), aa_frequencies.get('K', 0), aa_frequencies.get('R', 0),
            aa_frequencies.get('D', 0),
            # Additional features
            sequence.count('C') / length,  # Cysteine content (disulfide bonds)
            sequence.count('P') / length,  # Proline content (structural rigidity)
            sequence.count('G') / length,  # Glycine content (flexibility)
            sequence.count('W') / length + sequence.count('F') / length + sequence.count('Y') / length  # Aromatic content
        ]
        
        return np.array(features)

def load_and_preprocess_data(data_path: str, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the BRENDA training data."""
    logger.info(f"Loading data from {data_path}")
    
    # Load the data
    df = pd.read_csv(data_path, sep='\t')
    logger.info(f"Loaded {len(df)} records")
    
    # Remove rows with missing sequences or temperatures
    df = df.dropna(subset=['sequence', 'optimal_temp'])
    logger.info(f"After removing missing data: {len(df)} records")
    
    # Filter out sequences that are too short or too long
    df = df[(df['sequence'].str.len() >= 50) & (df['sequence'].str.len() <= 2000)]
    logger.info(f"After filtering sequence length: {len(df)} records")
    
    # Filter temperature range (reasonable enzyme temperatures)
    df = df[(df['optimal_temp'] >= 0) & (df['optimal_temp'] <= 150)]
    logger.info(f"After filtering temperature range: {len(df)} records")
    
    # Extract features
    feature_extractor = SequenceFeatureExtractor()
    logger.info("Extracting sequence features...")
    
    features = []
    targets = []
    
    for idx, row in df.iterrows():
        try:
            seq_features = feature_extractor.extract_features(row['sequence'])
            features.append(seq_features)
            targets.append(row['optimal_temp'])
        except Exception as e:
            logger.warning(f"Error processing sequence at index {idx}: {e}")
            continue
    
    X = np.array(features)
    y = np.array(targets)
    
    logger.info(f"Final dataset: {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Temperature range: {y.min():.1f}°C to {y.max():.1f}°C")
    logger.info(f"Mean temperature: {y.mean():.1f}°C ± {y.std():.1f}°C")
    
    return X, y

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray,
                       logger: logging.Logger) -> Tuple[Any, Dict[str, float]]:
    """Train and evaluate a Random Forest model."""
    logger.info("Training Random Forest model...")
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    logger.info(f"Best RF parameters: {grid_search.best_params_}")
    
    # Predictions
    y_pred = best_rf.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    logger.info(f"Random Forest - RMSE: {rmse:.2f}°C, MAE: {mae:.2f}°C, R²: {r2:.3f}")
    
    return best_rf, metrics

def train_svm(X_train: np.ndarray, y_train: np.ndarray, 
              X_test: np.ndarray, y_test: np.ndarray,
              logger: logging.Logger) -> Tuple[Any, Dict[str, float]]:
    """Train and evaluate an SVM model."""
    logger.info("Training SVM model...")
    
    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    
    svm = SVR()
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_svm = grid_search.best_estimator_
    logger.info(f"Best SVM parameters: {grid_search.best_params_}")
    
    # Predictions
    y_pred = best_svm.predict(X_test_scaled)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    logger.info(f"SVM - RMSE: {rmse:.2f}°C, MAE: {mae:.2f}°C, R²: {r2:.3f}")
    
    return (best_svm, scaler), metrics

def save_models(rf_model: Any, svm_model_scaler: Tuple[Any, Any], 
                output_dir: str, logger: logging.Logger):
    """Save trained models to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Random Forest
    rf_path = os.path.join(output_dir, 'random_forest_model.joblib')
    joblib.dump(rf_model, rf_path)
    logger.info(f"Random Forest model saved to {rf_path}")
    
    # Save SVM and scaler
    svm_model, scaler = svm_model_scaler
    svm_path = os.path.join(output_dir, 'svm_model.joblib')
    scaler_path = os.path.join(output_dir, 'svm_scaler.joblib')
    joblib.dump(svm_model, svm_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"SVM model saved to {svm_path}")
    logger.info(f"SVM scaler saved to {scaler_path}")

def main():
    parser = argparse.ArgumentParser(description="Train custom OptEnzML predictors")
    parser.add_argument('--data-path', default='data/raw/topt_data_final.tsv',
                       help='Path to training data file')
    parser.add_argument('--output-dir', default='data/models',
                       help='Directory to save trained models')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn is required for training. Please install it with: pip install scikit-learn")
        sys.exit(1)
    
    logger.info("Starting OptEnzML model training")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data(args.data_path, logger)
        
        if len(X) < 10:
            logger.error("Insufficient data for training (need at least 10 samples)")
            sys.exit(1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_seed
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train models
        rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test, logger)
        svm_model_scaler, svm_metrics = train_svm(X_train, y_train, X_test, y_test, logger)
        
        # Save models
        save_models(rf_model, svm_model_scaler, args.output_dir, logger)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        logger.info(f"Dataset: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"Temperature range: {y.min():.1f}°C - {y.max():.1f}°C")
        logger.info("")
        logger.info("Random Forest Performance:")
        logger.info(f"  RMSE: {rf_metrics['rmse']:.2f}°C")
        logger.info(f"  MAE:  {rf_metrics['mae']:.2f}°C")
        logger.info(f"  R²:   {rf_metrics['r2']:.3f}")
        logger.info("")
        logger.info("SVM Performance:")
        logger.info(f"  RMSE: {svm_metrics['rmse']:.2f}°C")
        logger.info(f"  MAE:  {svm_metrics['mae']:.2f}°C")
        logger.info(f"  R²:   {svm_metrics['r2']:.3f}")
        logger.info("="*60)
        
        # Save training report
        report_path = os.path.join(args.output_dir, 'training_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"OptEnzML Training Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data: {args.data_path}\n")
            f.write(f"Samples: {len(X)}\n")
            f.write(f"Features: {X.shape[1]}\n")
            f.write(f"Temperature range: {y.min():.1f}°C - {y.max():.1f}°C\n\n")
            f.write(f"Random Forest - RMSE: {rf_metrics['rmse']:.2f}°C, MAE: {rf_metrics['mae']:.2f}°C, R²: {rf_metrics['r2']:.3f}\n")
            f.write(f"SVM - RMSE: {svm_metrics['rmse']:.2f}°C, MAE: {svm_metrics['mae']:.2f}°C, R²: {svm_metrics['r2']:.3f}\n")
        
        logger.info(f"Training report saved to {report_path}")
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
