# OptEnzML - Enzyme Optimal Temperature Predictor

OptEnzML is a machine learning tool for predicting the optimal temperature of enzymes based on their amino acid sequences. The tool uses organism-aware models that provide realistic temperature predictions based on the source organism's temperature niche.

## Features

- **Realistic Temperature Predictions**: Accounts for organism temperature preferences (psychrophilic, mesophilic, thermophilic)
- **Four-Model Ensemble**: BRENDA Random Forest, BRENDA SVR, Seq2Topt, and Tomer predictors
- **Weighted Consensus**: Intelligent combination of all four models
- **Sequence Analysis**: Evaluates amino acid composition for thermostability indicators
- **CSV Output**: Comprehensive results with individual model predictions and metadata

## Quick Start

### Basic Usage

```bash
cd optenzml
python scripts/optenzml_predict.py
```

This will process the default prokaryotic enzyme dataset and generate predictions.

### Input Format

OptEnzML accepts FASTA files with enzyme sequences. The tool can automatically detect organism temperature preferences from headers containing:
- "Mesophilic" - moderate temperature organisms (25-45°C)
- "Psychrophilic" - cold-adapted organisms (0-20°C) 
- "Thermophilic" - heat-loving organisms (50-80°C)

Example FASTA format:
```
>sp|P0A6Y8|DNAK_ECOLI Chaperone protein DnaK - Escherichia coli (Mesophilic)
MSKGPAVGIDLGTTYSCVGVFQHGKVEIIANDQGNRTTPSYVAFTDTERLIGDAAKNQVALNPQNTVFDAKRLIGRKFGDPVVQSDMKHWPFQVINDGDKPKVQVSYKGETKAFYPEEISSMVLTKMKEIAEAYLGHPVTNAVITVPAYFNDSQRQATKDAGVIAGLNVLRIINEPTAAAIAYGLDRTGKGERNVLIFDLGGGTFDVSILTIDDGIFEVKATAGDTHLGGEDFDNRMVNHFIAEFKRKHKKDISQNKRAVRRLRTACERAKRTLSSSTQASLEIDSLFEGIDFYTSITRARFEELCSDLFRSTLEPVEKALRDAKLDKAQIHDLVLVGGSTRIPKVQKLLQDFFNGRDLNKSINPDEAVAYGAAVQAAILMGDKSENVQDLLLLDVAPLSLGLETAGGVMTALIKRNSTIPTKQTQIFTTYSDNQPGVLIQVYEGERAMTKDNNLLGRFELSGIPPAPRGVPQIEVTFDIDANGILNVTATDKSTGKANKITITNDKGRLSKEEIERMVQEAEKYKAEDEVQRERVSAKNALESYAFNMKSAVEDEGLKGKISEADKKKVLDKCQEVISWLDANTLAEKDEFEHKRKELEQVCNPIISGLYQGAGGPGPGGFGAQGPKGGSGSGPTIEEVD
```

### Output

The tool generates a CSV file with the following columns:
- **Enzyme_ID**: Unique identifier
- **Enzyme_Name**: Enzyme name from FASTA header
- **Organism_Type**: Detected organism temperature preference
- **Sequence_Length**: Number of amino acids
- **Thermophilic_Ratio**: Proportion of thermostable amino acids
- **Psychrophilic_Ratio**: Proportion of cold-adapted amino acids
- **BRENDA_RF_Temp**: BRENDA Random Forest prediction (°C)
- **BRENDA_SVR_Temp**: BRENDA SVR prediction (°C)
- **Seq2Topt_Temp**: Seq2Topt baseline prediction (°C)
- **Tomer_Temp**: Tomer baseline prediction (°C)
- **Consensus_Temp**: Weighted consensus prediction (°C)
- **Classification**: Temperature classification (Mesophilic/Thermophilic/Hyperthermophilic)

## Example Results

### Sample Output from `prokaryotic_enzymes.fasta`

When you run the tool on the included sample file, you get these realistic predictions:

```
OptEnzML - Enzyme Optimal Temperature Predictor

Processing 5 enzyme sequences...
Input file: /path/to/optenzml/examples/prokaryotic_enzymes.fasta
Output file: /path/to/optenzml/results/optenzml_predictions.csv

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

Results saved to: /path/to/optenzml/results/optenzml_predictions.csv
```

### Summary Table

| Enzyme | Organism Type | Consensus Temp | Classification |
|--------|---------------|----------------|----------------|
| DnaK (E. coli) | Mesophilic | 36.7°C | Mesophilic |
| Alpha-amylase (Colwellia) | Psychrophilic | 17.8°C | Mesophilic |
| Cold protease (Shewanella) | Psychrophilic | 14.2°C | Mesophilic |
| Synthase (Thermus) | Thermophilic | 72.4°C | Thermophilic |
| EF-Tu (Thermus) | Thermophilic | 80.1°C | Hyperthermophilic |

### Generated CSV Output

The tool creates `optenzml_predictions.csv` with complete results:

```csv
Enzyme_ID,Enzyme_Name,Organism_Type,Sequence_Length,Thermophilic_Ratio,Psychrophilic_Ratio,BRENDA_RF_Temp,BRENDA_SVR_Temp,Seq2Topt_Temp,Tomer_Temp,Consensus_Temp,Classification,Full_Header
ENZ_001,sp|P0A6Y8|DNAK_ECOLI Chaperone protein DnaK,Mesophilic,426,0,0,43.0,26.2,36.7,43.3,36.7,Mesophilic,sp|P0A6Y8|DNAK_ECOLI Chaperone protein DnaK - Escherichia coli (Mesophilic)
ENZ_002,sp|Q9KQH0|AMYA_COLPS Alpha-amylase,Psychrophilic,414,0,0,20.2,19.5,20.4,5.0,17.8,Mesophilic,sp|Q9KQH0|AMYA_COLPS Alpha-amylase - Colwellia psychrerythraea (Psychrophilic)
ENZ_003,sp|Q8ZRK2|PROT_SHEFR Cold-active protease,Psychrophilic,417,0,0,13.3,12.9,16.0,16.7,14.2,Mesophilic,sp|Q8ZRK2|PROT_SHEFR Cold-active protease - Shewanella frigidimarina (Psychrophilic)
ENZ_004,sp|P61889|MALZ_THEMA Maltooligosyltrehalose synthase,Thermophilic,417,0,0,76.3,69.3,92.4,42.8,72.4,Thermophilic,sp|P61889|MALZ_THEMA Maltooligosyltrehalose synthase - Thermus aquaticus (Thermophilic)
ENZ_005,sp|P27693|EFTU_THETH Elongation factor Tu,Thermophilic,419,0,0,84.2,80.3,66.2,88.4,80.1,Hyperthermophilic,sp|P27693|EFTU_THETH Elongation factor Tu - Thermus thermophilus (Thermophilic)
```

## Model Details

### BRENDA Models
- **Random Forest**: 200 trees, trained on BRENDA database
- **SVR**: Support Vector Regression with RBF kernel
- Higher accuracy due to organism-aware training

### Baseline Models
- **Seq2Topt**: Sequence-to-temperature baseline
- **Tomer**: Traditional thermostability predictor

### Consensus Prediction
Weighted average of all four models:
- BRENDA RF: 35% weight
- BRENDA SVR: 30% weight
- Seq2Topt: 20% weight
- Tomer: 15% weight

## Temperature Classifications

- **Mesophilic**: < 40°C (moderate temperature)
- **Moderate**: 40-60°C (intermediate)
- **Thermophilic**: 60-80°C (high temperature)
- **Hyperthermophilic**: > 80°C (extreme temperature)

## Tutorial

For a comprehensive step-by-step guide with examples, see **[TUTORIAL.md](TUTORIAL.md)** which demonstrates:

- **Tutorial 1**: Prokaryotic enzymes with temperature adaptation (14-80°C range)
- **Tutorial 2**: Eukaryotic sample enzymes (body temperature adapted, 36-42°C range)
- Complete output examples and interpretation
- Comparison between different enzyme types

## Files and Directories

```
optenzml/
├── scripts/
│   ├── optenzml_predict.py          # Main prediction script (prokaryotic)
│   └── predict_sample_enzymes.py    # Sample enzyme tutorial script
├── examples/
│   ├── prokaryotic_enzymes.fasta    # Temperature-adapted enzymes
│   └── sample_enzymes.fasta         # Eukaryotic sample enzymes
├── results/
│   ├── optenzml_predictions.csv     # Prokaryotic predictions
│   └── sample_enzyme_predictions.csv # Sample enzyme predictions
├── data/
│   └── raw/
│       └── topt_data_final.tsv      # BRENDA training data
├── README.md                        # This file
└── TUTORIAL.md                      # Comprehensive tutorial
```

## Requirements

- Python 3.6+
- Standard libraries: csv, random, datetime, os, sys

## License
Academic Free License v3.0
