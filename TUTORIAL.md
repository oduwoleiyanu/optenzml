# OptEnzML Tutorial - Predicting Enzyme Optimal Temperatures

This tutorial demonstrates how to use OptEnzML to predict optimal temperatures for different types of enzymes using the provided sample files.

## Overview

OptEnzML provides realistic temperature predictions using a four-model ensemble:
- **BRENDA Random Forest** (35% weight) - High accuracy model
- **BRENDA SVR** (30% weight) - Support Vector Regression model  
- **Seq2Topt** (20% weight) - Baseline sequence-to-temperature model
- **Tomer** (15% weight) - Traditional thermostability predictor

## Tutorial 1: Prokaryotic Enzymes (Temperature-Adapted)

### Input File: `prokaryotic_enzymes.fasta`

This file contains enzymes from organisms with different temperature preferences:
- **Psychrophilic** (cold-adapted): Colwellia, Shewanella
- **Mesophilic** (moderate temp): Escherichia coli  
- **Thermophilic** (heat-loving): Thermus species

### Running the Prediction

```bash
cd optenzml
python scripts/optenzml_predict.py
```

### Expected Output

```
============================================================
OptEnzML - Enzyme Optimal Temperature Predictor
============================================================

Processing 5 enzyme sequences...

Processing enzyme 1/5...
  Enzyme: sp|P0A6Y8|DNAK_ECOLI Chaperone protein DnaK
  Organism type: Mesophilic
  Sequence length: 426 amino acids
  - BRENDA RF: 43.0C
  - BRENDA SVR: 26.2C
  - Seq2Topt: 36.7C
  - Tomer: 43.3C
  - Consensus: 36.7C (Mesophilic)

Processing enzyme 2/5...
  Enzyme: sp|Q9KQH0|AMYA_COLPS Alpha-amylase
  Organism type: Psychrophilic
  Sequence length: 414 amino acids
  - BRENDA RF: 20.2C
  - BRENDA SVR: 19.5C
  - Seq2Topt: 20.4C
  - Tomer: 5.0C
  - Consensus: 17.8C (Mesophilic)

Processing enzyme 3/5...
  Enzyme: sp|Q8ZRK2|PROT_SHEFR Cold-active protease
  Organism type: Psychrophilic
  Sequence length: 417 amino acids
  - BRENDA RF: 13.3C
  - BRENDA SVR: 12.9C
  - Seq2Topt: 16.0C
  - Tomer: 16.7C
  - Consensus: 14.2C (Mesophilic)

Processing enzyme 4/5...
  Enzyme: sp|P61889|MALZ_THEMA Maltooligosyltrehalose synthase
  Organism type: Thermophilic
  Sequence length: 417 amino acids
  - BRENDA RF: 76.3C
  - BRENDA SVR: 69.3C
  - Seq2Topt: 92.4C
  - Tomer: 42.8C
  - Consensus: 72.4C (Thermophilic)

Processing enzyme 5/5...
  Enzyme: sp|P27693|EFTU_THETH Elongation factor Tu
  Organism type: Thermophilic
  Sequence length: 419 amino acids
  - BRENDA RF: 84.2C
  - BRENDA SVR: 80.3C
  - Seq2Topt: 66.2C
  - Tomer: 88.4C
  - Consensus: 80.1C (Hyperthermophilic)

PREDICTION SUMMARY
Total enzymes processed: 5
Successful predictions: 5
Average optimal temperature: 44.2C
Temperature range: 14.2C - 80.1C

Classification distribution:
  Hyperthermophilic: 1 enzymes
  Mesophilic: 3 enzymes
  Thermophilic: 1 enzymes
```

### Key Insights

- **Cold-adapted enzymes** (Colwellia, Shewanella): 14-18°C
- **Mesophilic enzyme** (E. coli): 37°C (body temperature)
- **Thermophilic enzymes** (Thermus): 72-80°C (high temperature)

## Tutorial 2: Eukaryotic Sample Enzymes

### Input File: `sample_enzymes.fasta`

This file contains enzymes from eukaryotic organisms (all mesophilic):
- **Chicken** (Lysozyme C) - body temp ~41°C
- **Human** (Calmodulin) - body temp ~37°C  
- **Pig** (Trypsin) - body temp ~39°C
- **E. coli** (Beta-galactosidase) - optimal ~37°C
- **Bovine** (Catalase) - body temp ~39°C

### Running the Prediction

```bash
cd optenzml
python scripts/predict_sample_enzymes.py
```

### Expected Output

```
======================================================================
OptEnzML - Sample Enzyme Prediction Tutorial
======================================================================

Processing 5 sample enzyme sequences...

Processing enzyme 1/5...
  Enzyme: sp|P00698|LYSC_CHICK Lysozyme C
  Organism: Chicken (body temp ~41°C)
  Sequence length: 147 amino acids
  Model Predictions:
    - BRENDA RF: 43.4°C
    - BRENDA SVR: 36.1°C
    - Seq2Topt: 39.8°C
    - Tomer: 43.2°C
  Consensus: 40.5°C (Moderate)

Processing enzyme 2/5...
  Enzyme: sp|P62593|CALM_HUMAN Calmodulin
  Organism: Human (body temp ~37°C)
  Sequence length: 149 amino acids
  Model Predictions:
    - BRENDA RF: 41.4°C
    - BRENDA SVR: 40.7°C
    - Seq2Topt: 41.6°C
    - Tomer: 23.7°C
  Consensus: 38.6°C (Mesophilic)

Processing enzyme 3/5...
  Enzyme: sp|P00761|TRYP_PIG Trypsin
  Organism: Pig (body temp ~39°C)
  Sequence length: 223 amino acids
  Model Predictions:
    - BRENDA RF: 37.4°C
    - BRENDA SVR: 36.9°C
    - Seq2Topt: 34.0°C
    - Tomer: 38.2°C
  Consensus: 36.7°C (Mesophilic)

Processing enzyme 4/5...
  Enzyme: sp|P00722|BGAL_ECOLI Beta-galactosidase
  Organism: E. coli (optimal ~37°C)
  Sequence length: 1024 amino acids
  Model Predictions:
    - BRENDA RF: 40.2°C
    - BRENDA SVR: 37.2°C
    - Seq2Topt: 45.6°C
    - Tomer: 18.2°C
  Consensus: 37.1°C (Mesophilic)

Processing enzyme 5/5...
  Enzyme: sp|P00805|CATA_BOVIN Catalase
  Organism: Bovine (body temp ~39°C)
  Sequence length: 8267 amino acids
  Model Predictions:
    - BRENDA RF: 43.9°C
    - BRENDA SVR: 42.1°C
    - Seq2Topt: 33.1°C
    - Tomer: 45.9°C
  Consensus: 41.5°C (Moderate)

SAMPLE ENZYME PREDICTION SUMMARY
Total enzymes processed: 5
Successful predictions: 5
Average optimal temperature: 38.9°C
Temperature range: 36.7°C - 41.5°C

Classification distribution:
  Mesophilic: 3 enzymes
  Moderate: 2 enzymes

Organism distribution:
  Bovine (body temp ~39°C): 1 enzymes
  Chicken (body temp ~41°C): 1 enzymes
  E. coli (optimal ~37°C): 1 enzymes
  Human (body temp ~37°C): 1 enzymes
  Pig (body temp ~39°C): 1 enzymes
```

### Key Insights

- **All predictions are realistic** (36-42°C range)
- **Body temperature correlation**: Predictions align with organism body temperatures
- **Narrow range**: Eukaryotic enzymes show less temperature variation than prokaryotic

## Comparison: Prokaryotic vs Eukaryotic Enzymes

| Dataset | Temperature Range | Average | Key Features |
|---------|------------------|---------|--------------|
| **Prokaryotic** | 14.2°C - 80.1°C | 44.2°C | Wide range, includes extremophiles |
| **Eukaryotic** | 36.7°C - 41.5°C | 38.9°C | Narrow range, body temperature adapted |

## Understanding the Output Files

### CSV Output Columns

Both scripts generate comprehensive CSV files with these columns:

- **Enzyme_ID**: Unique identifier (ENZ_001, SAMPLE_001, etc.)
- **Enzyme_Name**: Extracted enzyme name
- **Organism_Type/Organism_Info**: Temperature preference or organism details
- **Sequence_Length**: Number of amino acids
- **Thermophilic_Ratio**: Proportion of heat-stable amino acids
- **Psychrophilic_Ratio**: Proportion of cold-adapted amino acids
- **BRENDA_RF_Temp**: Random Forest prediction (°C)
- **BRENDA_SVR_Temp**: SVR prediction (°C)
- **Seq2Topt_Temp**: Baseline model prediction (°C)
- **Tomer_Temp**: Traditional model prediction (°C)
- **Consensus_Temp**: Weighted average prediction (°C)
- **Classification**: Temperature category
- **Full_Header**: Complete FASTA header

### Temperature Classifications

- **Mesophilic**: < 40°C (moderate temperature)
- **Moderate**: 40-60°C (intermediate)
- **Thermophilic**: 60-80°C (high temperature)
- **Hyperthermophilic**: > 80°C (extreme temperature)

## Model Performance Insights

### BRENDA Models (Higher Accuracy)
- **Random Forest**: Best overall performance, organism-aware
- **SVR**: Consistent predictions, good for consensus

### Baseline Models
- **Seq2Topt**: Sequence-based, moderate accuracy
- **Tomer**: Traditional approach, lower weight in consensus

### Consensus Strategy
The weighted consensus provides the most reliable predictions by combining all four models with weights based on their expected performance.

## Tips for Using OptEnzML

1. **Organism Information**: Include organism names in FASTA headers for better predictions
2. **Temperature Hints**: Add "(Mesophilic)", "(Thermophilic)", etc. to headers when known
3. **Sequence Quality**: Ensure complete, high-quality protein sequences
4. **Result Interpretation**: Focus on consensus predictions for most reliable results
5. **Comparative Analysis**: Use multiple enzymes to understand temperature patterns

## Next Steps

- Try your own enzyme sequences
- Compare predictions across different organisms
- Analyze the relationship between sequence features and temperature predictions
- Use the CSV output for further statistical analysis

## Files Generated

- `optenzml_predictions.csv` - Prokaryotic enzyme results
- `sample_enzyme_predictions.csv` - Eukaryotic enzyme results

Both files are saved in the `optenzml/results/` directory and can be opened in Excel or analyzed with Python/R for further insights.
