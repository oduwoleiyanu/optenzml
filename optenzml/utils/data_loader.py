"""
Data loading utilities for OptEnzML

This module provides utilities for loading protein sequences from various formats
and validating sequence data.
"""

import os
import logging
from typing import List, Tuple, Optional, Dict, Any, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    logger.warning("BioPython not available. Using basic FASTA parser.")


class SequenceValidator:
    """
    Utility class for validating protein sequences.
    """
    
    def __init__(self):
        self.valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        self.ambiguous_amino_acids = set('XBZJUO')
        self.all_valid = self.valid_amino_acids | self.ambiguous_amino_acids
    
    def is_valid_protein_sequence(self, sequence: str) -> bool:
        """
        Check if a sequence is a valid protein sequence.
        
        Args:
            sequence: Protein sequence to validate
            
        Returns:
            True if sequence is valid, False otherwise
        """
        if not sequence or not isinstance(sequence, str):
            return False
        
        # Remove whitespace and convert to uppercase
        clean_sequence = sequence.strip().upper()
        
        if len(clean_sequence) == 0:
            return False
        
        # Check if all characters are valid amino acids
        sequence_chars = set(clean_sequence)
        return sequence_chars.issubset(self.all_valid)
    
    def validate_sequence_length(self, sequence: str, min_length: int = 10, 
                               max_length: int = 10000) -> bool:
        """
        Validate sequence length.
        
        Args:
            sequence: Protein sequence
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            
        Returns:
            True if length is valid, False otherwise
        """
        if not sequence:
            return False
        
        length = len(sequence.strip())
        return min_length <= length <= max_length
    
    def get_sequence_composition(self, sequence: str) -> Dict[str, float]:
        """
        Get amino acid composition of a sequence.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Dictionary with amino acid frequencies
        """
        if not sequence:
            return {}
        
        clean_sequence = sequence.strip().upper()
        length = len(clean_sequence)
        
        if length == 0:
            return {}
        
        composition = {}
        for aa in self.valid_amino_acids:
            count = clean_sequence.count(aa)
            composition[aa] = count / length
        
        # Count ambiguous amino acids
        for aa in self.ambiguous_amino_acids:
            count = clean_sequence.count(aa)
            if count > 0:
                composition[aa] = count / length
        
        return composition
    
    def clean_sequence(self, sequence: str) -> str:
        """
        Clean and standardize a protein sequence.
        
        Args:
            sequence: Raw protein sequence
            
        Returns:
            Cleaned sequence
        """
        if not sequence:
            return ""
        
        # Remove whitespace and convert to uppercase
        clean_sequence = sequence.strip().upper()
        
        # Remove any non-amino acid characters
        clean_sequence = ''.join(char for char in clean_sequence if char in self.all_valid)
        
        return clean_sequence


class DataLoader:
    """
    Utility class for loading protein sequences from various file formats.
    """
    
    def __init__(self):
        self.validator = SequenceValidator()
        self.supported_formats = ['fasta', 'fa', 'fas', 'txt']
    
    def load_fasta_file(self, filepath: str) -> List[Tuple[str, str]]:
        """
        Load sequences from a FASTA file.
        
        Args:
            filepath: Path to FASTA file
            
        Returns:
            List of (header, sequence) tuples
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        sequences = []
        
        if BIOPYTHON_AVAILABLE:
            sequences = self._load_fasta_biopython(filepath)
        else:
            sequences = self._load_fasta_basic(filepath)
        
        logger.info(f"Loaded {len(sequences)} sequences from {filepath}")
        return sequences
    
    def _load_fasta_biopython(self, filepath: str) -> List[Tuple[str, str]]:
        """Load FASTA file using BioPython."""
        sequences = []
        
        try:
            for record in SeqIO.parse(filepath, "fasta"):
                header = record.description
                sequence = str(record.seq)
                
                if self.validator.is_valid_protein_sequence(sequence):
                    sequences.append((header, sequence))
                else:
                    logger.warning(f"Invalid sequence skipped: {header[:50]}...")
        
        except Exception as e:
            logger.error(f"Error parsing FASTA file with BioPython: {e}")
            # Fallback to basic parser
            sequences = self._load_fasta_basic(filepath)
        
        return sequences
    
    def _load_fasta_basic(self, filepath: str) -> List[Tuple[str, str]]:
        """Load FASTA file using basic parser."""
        sequences = []
        current_header = None
        current_sequence = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    if line.startswith('>'):
                        # Save previous sequence if exists
                        if current_header is not None and current_sequence:
                            sequence = ''.join(current_sequence)
                            if self.validator.is_valid_protein_sequence(sequence):
                                sequences.append((current_header, sequence))
                            else:
                                logger.warning(f"Invalid sequence skipped at line {line_num}: {current_header[:50]}...")
                        
                        # Start new sequence
                        current_header = line[1:]  # Remove '>'
                        current_sequence = []
                    
                    else:
                        # Sequence line
                        if current_header is None:
                            logger.warning(f"Sequence data without header at line {line_num}")
                            continue
                        current_sequence.append(line)
                
                # Save last sequence
                if current_header is not None and current_sequence:
                    sequence = ''.join(current_sequence)
                    if self.validator.is_valid_protein_sequence(sequence):
                        sequences.append((current_header, sequence))
                    else:
                        logger.warning(f"Invalid sequence skipped: {current_header[:50]}...")
        
        except Exception as e:
            logger.error(f"Error parsing FASTA file: {e}")
            raise
        
        return sequences
    
    def load_sequences_from_directory(self, directory: str, 
                                    recursive: bool = False) -> List[Tuple[str, str, str]]:
        """
        Load sequences from all FASTA files in a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively
            
        Returns:
            List of (filename, header, sequence) tuples
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")
        
        all_sequences = []
        fasta_files = self.find_fasta_files(directory, recursive)
        
        for filepath in fasta_files:
            try:
                sequences = self.load_fasta_file(filepath)
                filename = os.path.basename(filepath)
                
                for header, sequence in sequences:
                    all_sequences.append((filename, header, sequence))
            
            except Exception as e:
                logger.error(f"Error loading file {filepath}: {e}")
                continue
        
        logger.info(f"Loaded {len(all_sequences)} sequences from {len(fasta_files)} files")
        return all_sequences
    
    def find_fasta_files(self, directory: str, recursive: bool = False) -> List[str]:
        """
        Find all FASTA files in a directory.
        
        Args:
            directory: Directory to search
            recursive: Whether to search recursively
            
        Returns:
            List of FASTA file paths
        """
        fasta_files = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if self._is_fasta_file(file):
                        fasta_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                filepath = os.path.join(directory, file)
                if os.path.isfile(filepath) and self._is_fasta_file(file):
                    fasta_files.append(filepath)
        
        return sorted(fasta_files)
    
    def _is_fasta_file(self, filename: str) -> bool:
        """Check if a file is likely a FASTA file based on extension."""
        extension = Path(filename).suffix.lower().lstrip('.')
        return extension in self.supported_formats
    
    def save_sequences_to_fasta(self, sequences: List[Tuple[str, str]], 
                               output_path: str) -> None:
        """
        Save sequences to a FASTA file.
        
        Args:
            sequences: List of (header, sequence) tuples
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for header, sequence in sequences:
                    # Ensure header starts with '>'
                    if not header.startswith('>'):
                        header = '>' + header
                    
                    f.write(f"{header}\n")
                    
                    # Write sequence in lines of 80 characters
                    for i in range(0, len(sequence), 80):
                        f.write(f"{sequence[i:i+80]}\n")
            
            logger.info(f"Saved {len(sequences)} sequences to {output_path}")
        
        except Exception as e:
            logger.error(f"Error saving sequences to {output_path}: {e}")
            raise
    
    def load_sequence_from_string(self, sequence_string: str) -> str:
        """
        Load and validate a sequence from a string.
        
        Args:
            sequence_string: Raw sequence string
            
        Returns:
            Cleaned and validated sequence
        """
        clean_sequence = self.validator.clean_sequence(sequence_string)
        
        if not self.validator.is_valid_protein_sequence(clean_sequence):
            raise ValueError("Invalid protein sequence")
        
        return clean_sequence
    
    def batch_validate_sequences(self, sequences: List[str]) -> List[Tuple[int, bool, str]]:
        """
        Validate multiple sequences.
        
        Args:
            sequences: List of sequences to validate
            
        Returns:
            List of (index, is_valid, error_message) tuples
        """
        results = []
        
        for i, sequence in enumerate(sequences):
            try:
                is_valid = self.validator.is_valid_protein_sequence(sequence)
                error_msg = "" if is_valid else "Invalid amino acid characters"
                
                if is_valid and not self.validator.validate_sequence_length(sequence):
                    is_valid = False
                    error_msg = "Sequence length out of range"
                
                results.append((i, is_valid, error_msg))
            
            except Exception as e:
                results.append((i, False, str(e)))
        
        return results
    
    def get_sequence_statistics(self, sequences: List[str]) -> Dict[str, Any]:
        """
        Get statistics for a collection of sequences.
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            Dictionary with sequence statistics
        """
        if not sequences:
            return {}
        
        lengths = [len(seq) for seq in sequences]
        valid_sequences = [seq for seq in sequences if self.validator.is_valid_protein_sequence(seq)]
        
        stats = {
            'total_sequences': len(sequences),
            'valid_sequences': len(valid_sequences),
            'invalid_sequences': len(sequences) - len(valid_sequences),
            'length_stats': {
                'min': min(lengths) if lengths else 0,
                'max': max(lengths) if lengths else 0,
                'mean': sum(lengths) / len(lengths) if lengths else 0,
                'median': sorted(lengths)[len(lengths) // 2] if lengths else 0
            }
        }
        
        # Amino acid composition statistics
        if valid_sequences:
            all_compositions = [self.validator.get_sequence_composition(seq) for seq in valid_sequences]
            
            # Average composition
            avg_composition = {}
            for aa in self.validator.valid_amino_acids:
                values = [comp.get(aa, 0) for comp in all_compositions]
                avg_composition[aa] = sum(values) / len(values) if values else 0
            
            stats['average_composition'] = avg_composition
        
        return stats
