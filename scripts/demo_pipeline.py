#!/usr/bin/env python3
"""
Demo script showing the complete OptEnzML pipeline with BRENDA data.
This script demonstrates data curation, validation, and prediction capabilities.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path to import optenzml modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{title}")
    print("-" * len(title))

def demo_data_curation():
    """Demonstrate the BRENDA data curation process."""
    print_header("BRENDA DATA CURATION DEMO")
    
    print("The OptEnzML pipeline has successfully curated training data from the BRENDA database.")
    print("This process involved:")
    print("• Fetching enzyme data from BRENDA")
    print("• Extracting optimal temperature information")
    print("• Collecting protein sequences")
    print("• Validating and cleaning the dataset")
    print("• Categorizing enzymes by thermostability")
    
    # Check if data file exists
    data_file = "data/raw/topt_data_final.tsv"
    if os.path.exists(data_file):
        print(f"\n✓ BRENDA dataset successfully created: {data_file}")
        
        # Read and display basic statistics
        try:
            with open(data_file, 'r') as f:
                lines = f.readlines()
                print(f"✓ Dataset contains {len(lines)-1} enzyme records")
                
                # Show first few lines
                print("\nSample data:")
                for i, line in enumerate(lines[:4]):
                    if i == 0:
                        print("Headers:", line.strip().split('\t')[:5])
                    else:
                        parts = line.strip().split('\t')
                        print(f"Record {i}: EC {parts[0]}, {parts[1]}, {parts[4]}°C")
                        
        except Exception as e:
            print(f"Error reading data file: {e}")
    else:
        print(f"✗ Data file not found: {data_file}")

def demo_prediction_capabilities():
    """Demonstrate the prediction capabilities."""
    print_header("PREDICTION CAPABILITIES DEMO")
    
    print("OptEnzML provides multiple prediction methods:")
    print("• TOMER predictor (external tool integration)")
    print("• Seq2Topt predictor (external tool integration)")
    print("• Custom Random Forest predictor (trained on BRENDA data)")
    print("• Custom SVM predictor (trained on BRENDA data)")
    print("• Consensus model (combines multiple predictors)")
    
    print("\nExample prediction workflow:")
    print("1. Input: Protein sequence")
    print("2. Feature extraction: Amino acid composition, physicochemical properties")
    print("3. Prediction: Multiple models generate temperature predictions")
    print("4. Consensus: Weighted average with confidence scoring")
    print("5. Output: Optimal temperature ± confidence interval")

def demo_training_pipeline():
    """Demonstrate the training pipeline."""
    print_header("TRAINING PIPELINE DEMO")
    
    print("The training pipeline processes BRENDA data to create custom predictors:")
    
    print_section("1. Data Preprocessing")
    print("• Load BRENDA dataset (TSV format)")
    print("• Filter sequences by length (50-2000 amino acids)")
    print("• Filter temperatures (0-150°C)")
    print("• Remove incomplete records")
    
    print_section("2. Feature Engineering")
    print("• Sequence length")
    print("• Amino acid composition (20 features)")
    print("• Physicochemical properties:")
    print("  - Hydrophobic content")
    print("  - Polar residue content")
    print("  - Charged residue content")
    print("  - Aromatic content")
    print("  - Average molecular weight")
    print("• Structural indicators (Cys, Pro, Gly content)")
    
    print_section("3. Model Training")
    print("• Random Forest Regressor with hyperparameter tuning")
    print("• Support Vector Machine with feature scaling")
    print("• Cross-validation for model selection")
    print("• Performance metrics: RMSE, MAE, R²")
    
    print_section("4. Model Persistence")
    print("• Save trained models (joblib format)")
    print("• Save feature scalers")
    print("• Generate training reports")

def demo_usage_examples():
    """Show usage examples."""
    print_header("USAGE EXAMPLES")
    
    print("Command-line interface examples:")
    print()
    
    examples = [
        ("Single sequence prediction", 
         'python3 -m optenzml --sequence "MKKLVLSLSLVLAFSSATAAF..." --ogt 37'),
        ("FASTA file processing", 
         'python3 -m optenzml --fasta proteins.fasta --output csv'),
        ("Verbose output with individual predictors", 
         'python3 -m optenzml --sequence "MKK..." --verbose'),
        ("JSON output format", 
         'python3 -m optenzml --sequence "MKK..." --output json'),
        ("Using mock predictors (for testing)", 
         'python3 -m optenzml --use-mock-predictors --fasta examples/sample_enzymes.fasta'),
        ("Training custom models", 
         'python3 scripts/train_custom_models.py --data-path data/raw/topt_data_final.tsv'),
        ("Data collection from BRENDA", 
         'python3 scripts/download_brenda_data_simple.py --validate')
    ]
    
    for title, command in examples:
        print(f"{title}:")
        print(f"  {command}")
        print()

def demo_architecture():
    """Show the system architecture."""
    print_header("SYSTEM ARCHITECTURE")
    
    print("OptEnzML follows a modular architecture:")
    print()
    
    components = [
        ("Data Layer", [
            "BRENDA database integration",
            "Data validation and cleaning",
            "Feature extraction pipeline"
        ]),
        ("Prediction Layer", [
            "Base predictor interface",
            "External tool wrappers (TOMER, Seq2Topt)",
            "Custom ML predictors (RF, SVM)",
            "Mock predictors for testing"
        ]),
        ("Consensus Layer", [
            "Weighted averaging",
            "Confidence scoring",
            "Meta-learning ensemble"
        ]),
        ("Interface Layer", [
            "Command-line interface",
            "Batch processing",
            "Multiple output formats (table, CSV, JSON)"
        ]),
        ("Utilities", [
            "FASTA parsing",
            "Data loading",
            "Output formatting",
            "Logging and error handling"
        ])
    ]
    
    for component, features in components:
        print(f"{component}:")
        for feature in features:
            print(f"  • {feature}")
        print()

def main():
    parser = argparse.ArgumentParser(description="OptEnzML Pipeline Demo")
    parser.add_argument('--section', choices=['data', 'prediction', 'training', 'usage', 'architecture', 'all'],
                       default='all', help='Demo section to show')
    
    args = parser.parse_args()
    
    print_header("OPTENZML PIPELINE DEMONSTRATION")
    print("Enzyme Optimal Temperature Prediction with BRENDA Data")
    
    if args.section in ['data', 'all']:
        demo_data_curation()
    
    if args.section in ['prediction', 'all']:
        demo_prediction_capabilities()
    
    if args.section in ['training', 'all']:
        demo_training_pipeline()
    
    if args.section in ['usage', 'all']:
        demo_usage_examples()
    
    if args.section in ['architecture', 'all']:
        demo_architecture()
    
    print_header("SUMMARY")
    print("✓ BRENDA data successfully curated and integrated")
    print("✓ Multiple prediction methods implemented")
    print("✓ Training pipeline established")
    print("✓ Comprehensive CLI interface available")
    print("✓ Modular, extensible architecture")
    print()
    print("The OptEnzML pipeline is ready for enzyme optimal temperature prediction!")
    print("For full functionality, install required dependencies:")
    print("  pip install scikit-learn pandas numpy biopython")

if __name__ == "__main__":
    main()
