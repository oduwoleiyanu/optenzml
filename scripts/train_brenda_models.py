#!/usr/bin/env python3
"""
Train SVR and Random Forest models on BRENDA data with dipeptide features.
This script loads the actual topt_data_final.tsv and trains models with comprehensive features.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

# Add the parent directory to the path to import optenzml modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn not available. Please install with: pip install scikit-learn")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrendaFeatureExtractor:
    """Extract comprehensive features from protein sequences for BRENDA-trained models."""
    
    def __init__(self):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        # Amino acid properties
        self.aa_properties = {
            'A': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 89.1},
            'R': {'hydrophobic': 0, 'polar': 1, 'charged': 1, 'aromatic': 0, 'mw': 174.2},
            'N': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 0, 'mw': 132.1},
            'D': {'hydrophobic': 0, 'polar': 1, 'charged': -1, 'aromatic': 0, 'mw': 133.1},
            'C': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 121.2},
            'Q': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 0, 'mw': 146.1},
            'E': {'hydrophobic': 0, 'polar': 1, 'charged': -1, 'aromatic': 0, 'mw': 147.1},
            'G': {'hydrophobic': 0, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 75.1},
            'H': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 1, 'mw': 155.2},
            'I': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 131.2},
            'L': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 131.2},
            'K': {'hydrophobic': 0, 'polar': 1, 'charged': 1, 'aromatic': 0, 'mw': 146.2},
            'M': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 149.2},
            'F': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 1, 'mw': 165.2},
            'P': {'hydrophobic': 0, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 115.1},
            'S': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 0, 'mw': 105.1},
            'T': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 0, 'mw': 119.1},
            'W': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 1, 'mw': 204.2},
            'Y': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 1, 'mw': 181.2},
            'V': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 117.1}
        }
        
        # Generate all possible dipeptides
        self.all_dipeptides = [aa1 + aa2 for aa1 in self.amino_acids for aa2 in self.amino_acids]
        
    def extract_features(self, sequence: str) -> np.ndarray:
        """Extract comprehensive features from a protein sequence."""
        sequence = sequence.upper().strip()
        length = len(sequence)
        
        if length == 0:
            return np.zeros(self._get_feature_count())
        
        features = []
        
        # 1. Basic sequence properties
        features.append(length)
        
        # 2. Amino acid composition (20 features)
        aa_counts = {aa: sequence.count(aa) for aa in self.amino_acids}
        aa_frequencies = {aa: count/length for aa, count in aa_counts.items()}
        features.extend([aa_frequencies[aa] for aa in self.amino_acids])
        
        # 3. Dipeptide composition (400 features)
        dipeptide_counts = {}
        for i in range(length - 1):
            dipep = sequence[i:i+2]
            if len(dipep) == 2 and all(aa in self.amino_acids for aa in dipep):
                dipeptide_counts[dipep] = dipeptide_counts.get(dipep, 0) + 1
        
        # Normalize dipeptide counts
        total_dipeptides = max(1, length - 1)
        for dipep in self.all_dipeptides:
            count = dipeptide_counts.get(dipep, 0)
            features.append(count / total_dipeptides)
        
        # 4. Physicochemical properties
        # Hydrophobic content
        hydrophobic_count = sum(1 for aa in sequence if self.aa_properties.get(aa, {}).get('hydrophobic', 0))
        features.append(hydrophobic_count / length)
        
        # Polar content
        polar_count = sum(1 for aa in sequence if self.aa_properties.get(aa, {}).get('polar', 0))
        features.append(polar_count / length)
        
        # Charged content
        charged_count = sum(1 for aa in sequence if abs(self.aa_properties.get(aa, {}).get('charged', 0)) > 0)
        features.append(charged_count / length)
        
        # Aromatic content
        aromatic_count = sum(1 for aa in sequence if self.aa_properties.get(aa, {}).get('aromatic', 0))
        features.append(aromatic_count / length)
        
        # Average molecular weight
        total_mw = sum(self.aa_properties.get(aa, {}).get('mw', 0) for aa in sequence)
        features.append(total_mw / length)
        
        # 5. Structural indicators
        features.append(sequence.count('C') / length)  # Cysteine (disulfide bonds)
        features.append(sequence.count('P') / length)  # Proline (rigidity)
        features.append(sequence.count('G') / length)  # Glycine (flexibility)
        
        # 6. Thermostability indicators (based on literature)
        thermophilic_aa = set('GPRE')  # Common in thermophiles
        mesophilic_aa = set('QNST')    # Common in mesophiles
        features.append(sum(aa_frequencies.get(aa, 0) for aa in thermophilic_aa))
        features.append(sum(aa_frequencies.get(aa, 0) for aa in mesophilic_aa))
        
        # 7. Charge distribution
        positive_aa = set('RK')
        negative_aa = set('DE')
        features.append(sum(aa_frequencies.get(aa, 0) for aa in positive_aa))
        features.append(sum(aa_frequencies.get(aa, 0) for aa in negative_aa))
        
        return np.array(features)
    
    def _get_feature_count(self) -> int:
        """Get total number of features."""
        return 1 + 20 + 400 + 5 + 3 + 2 + 2  # 433 features total
    
    def get_feature_names(self) -> list:
        """Get names of all features."""
        names = ['length']
        names.extend([f'aa_{aa}' for aa in self.amino_acids])
        names.extend([f'dipep_{dipep}' for dipep in self.all_dipeptides])
        names.extend(['hydrophobic_ratio', 'polar_ratio', 'charged_ratio', 'aromatic_ratio', 'avg_mw'])
        names.extend(['cys_ratio', 'pro_ratio', 'gly_ratio'])
        names.extend(['thermophilic_ratio', 'mesophilic_ratio'])
        names.extend(['positive_ratio', 'negative_ratio'])
        return names

def load_brenda_data(data_path: str) -> tuple:
    """Load and preprocess BRENDA data."""
    logger.info(f"Loading BRENDA data from {data_path}")
    
    df = pd.read_csv(data_path, sep='\t')
    logger.info(f"Loaded {len(df)} enzyme records")
    
    # Extract sequences and temperatures
    sequences = df['sequence'].tolist()
    temperatures = df['optimal_temp'].tolist()
    
    # Additional metadata
    organisms = df['organism'].tolist()
    ec_numbers = df['ec_number'].tolist()
    categories = df['thermostability_category'].tolist()
    
    logger.info(f"Temperature range: {min(temperatures):.1f}°C - {max(temperatures):.1f}°C")
    logger.info(f"Average temperature: {np.mean(temperatures):.1f}°C")
    
    # Count categories
    category_counts = pd.Series(categories).value_counts()
    logger.info(f"Thermostability categories: {dict(category_counts)}")
    
    return sequences, temperatures, organisms, ec_numbers, categories

def train_models(X: np.ndarray, y: np.ndarray) -> tuple:
    """Train Random Forest and SVR models."""
    logger.info("Training models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Scale features for SVR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    logger.info("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Train SVR
    logger.info("Training SVR...")
    svr_model = SVR(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        epsilon=0.1
    )
    svr_model.fit(X_train_scaled, y_train)
    
    # Evaluate models
    logger.info("Evaluating models...")
    
    # Random Forest evaluation
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)
    
    rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
    rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
    rf_train_r2 = r2_score(y_train, rf_train_pred)
    rf_test_r2 = r2_score(y_test, rf_test_pred)
    rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
    
    logger.info(f"Random Forest - Train RMSE: {rf_train_rmse:.2f}, Test RMSE: {rf_test_rmse:.2f}")
    logger.info(f"Random Forest - Train R²: {rf_train_r2:.3f}, Test R²: {rf_test_r2:.3f}")
    logger.info(f"Random Forest - Test MAE: {rf_test_mae:.2f}")
    
    # SVR evaluation
    svr_train_pred = svr_model.predict(X_train_scaled)
    svr_test_pred = svr_model.predict(X_test_scaled)
    
    svr_train_rmse = np.sqrt(mean_squared_error(y_train, svr_train_pred))
    svr_test_rmse = np.sqrt(mean_squared_error(y_test, svr_test_pred))
    svr_train_r2 = r2_score(y_train, svr_train_pred)
    svr_test_r2 = r2_score(y_test, svr_test_pred)
    svr_test_mae = mean_absolute_error(y_test, svr_test_pred)
    
    logger.info(f"SVR - Train RMSE: {svr_train_rmse:.2f}, Test RMSE: {svr_test_rmse:.2f}")
    logger.info(f"SVR - Train R²: {svr_train_r2:.3f}, Test R²: {svr_test_r2:.3f}")
    logger.info(f"SVR - Test MAE: {svr_test_mae:.2f}")
    
    # Cross-validation
    logger.info("Performing cross-validation...")
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    svr_cv_scores = cross_val_score(svr_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    
    rf_cv_rmse = np.sqrt(-rf_cv_scores.mean())
    svr_cv_rmse = np.sqrt(-svr_cv_scores.mean())
    
    logger.info(f"Random Forest - CV RMSE: {rf_cv_rmse:.2f} ± {np.sqrt(-rf_cv_scores).std():.2f}")
    logger.info(f"SVR - CV RMSE: {svr_cv_rmse:.2f} ± {np.sqrt(-svr_cv_scores).std():.2f}")
    
    # Feature importance for Random Forest
    feature_importance = rf_model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:][::-1]
    
    logger.info("Top 10 most important features (Random Forest):")
    feature_extractor = BrendaFeatureExtractor()
    feature_names = feature_extractor.get_feature_names()
    for i, idx in enumerate(top_features_idx):
        logger.info(f"{i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    return rf_model, svr_model, scaler, {
        'rf_test_rmse': rf_test_rmse,
        'rf_test_r2': rf_test_r2,
        'rf_test_mae': rf_test_mae,
        'svr_test_rmse': svr_test_rmse,
        'svr_test_r2': svr_test_r2,
        'svr_test_mae': svr_test_mae,
        'rf_cv_rmse': rf_cv_rmse,
        'svr_cv_rmse': svr_cv_rmse
    }

def save_models(rf_model, svr_model, scaler, metrics, output_dir: str):
    """Save trained models and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Random Forest model
    rf_path = os.path.join(output_dir, 'brenda_random_forest.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump({
            'model': rf_model,
            'model_type': 'RandomForest',
            'training_data': 'BRENDA',
            'metrics': {k: v for k, v in metrics.items() if k.startswith('rf_')},
            'feature_count': 433
        }, f)
    logger.info(f"Saved Random Forest model to {rf_path}")
    
    # Save SVR model
    svr_path = os.path.join(output_dir, 'brenda_svr.pkl')
    with open(svr_path, 'wb') as f:
        pickle.dump({
            'model': svr_model,
            'scaler': scaler,
            'model_type': 'SVR',
            'training_data': 'BRENDA',
            'metrics': {k: v for k, v in metrics.items() if k.startswith('svr_')},
            'feature_count': 433
        }, f)
    logger.info(f"Saved SVR model to {svr_path}")
    
    # Save training report
    report_path = os.path.join(output_dir, 'brenda_training_report.txt')
    with open(report_path, 'w') as f:
        f.write("BRENDA Model Training Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Date: {pd.Timestamp.now()}\n")
        f.write(f"Training Data: BRENDA enzyme database\n")
        f.write(f"Feature Count: 433 (length + 20 AA + 400 dipeptides + 12 physicochemical)\n\n")
        
        f.write("Random Forest Results:\n")
        f.write(f"  Test RMSE: {metrics['rf_test_rmse']:.2f}°C\n")
        f.write(f"  Test R²: {metrics['rf_test_r2']:.3f}\n")
        f.write(f"  Test MAE: {metrics['rf_test_mae']:.2f}°C\n")
        f.write(f"  CV RMSE: {metrics['rf_cv_rmse']:.2f}°C\n\n")
        
        f.write("SVR Results:\n")
        f.write(f"  Test RMSE: {metrics['svr_test_rmse']:.2f}°C\n")
        f.write(f"  Test R²: {metrics['svr_test_r2']:.3f}\n")
        f.write(f"  Test MAE: {metrics['svr_test_mae']:.2f}°C\n")
        f.write(f"  CV RMSE: {metrics['svr_cv_rmse']:.2f}°C\n")
    
    logger.info(f"Saved training report to {report_path}")

def main():
    """Main training function."""
    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_path = project_dir / "data" / "raw" / "topt_data_final.tsv"
    models_dir = project_dir / "data" / "models"
    
    if not data_path.exists():
        logger.error(f"BRENDA data file not found: {data_path}")
        sys.exit(1)
    
    # Load data
    sequences, temperatures, organisms, ec_numbers, categories = load_brenda_data(str(data_path))
    
    # Extract features
    logger.info("Extracting features...")
    feature_extractor = BrendaFeatureExtractor()
    
    X = []
    for i, seq in enumerate(sequences):
        if i % 10 == 0:
            logger.info(f"Processing sequence {i+1}/{len(sequences)}")
        features = feature_extractor.extract_features(seq)
        X.append(features)
    
    X = np.array(X)
    y = np.array(temperatures)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target vector shape: {y.shape}")
    
    # Train models
    rf_model, svr_model, scaler, metrics = train_models(X, y)
    
    # Save models
    save_models(rf_model, svr_model, scaler, metrics, str(models_dir))
    
    logger.info("Training completed successfully!")
    logger.info(f"Models saved to: {models_dir}")

if __name__ == "__main__":
    main()
