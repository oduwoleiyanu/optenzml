"""
Command-line interface for OptEnzML

This module provides the main CLI functionality for the OptEnzML enzyme
optimal temperature prediction tool.
"""

import argparse
import sys
import os
import logging
from typing import List, Optional
from pathlib import Path

from ..predictors.tomer_predictor import TomerPredictor, MockTomerPredictor
from ..predictors.seq2topt_predictor import Seq2ToptPredictor, MockSeq2ToptPredictor
from ..predictors.custom_predictor import CustomRFPredictor, CustomSVMPredictor
from ..consensus.consensus_model import ConsensusModel
from ..utils.data_loader import DataLoader
from ..utils.output_formatter import OutputFormatter


class OptEnzMLCLI:
    """
    Command-line interface for OptEnzML.
    """
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.output_formatter = OutputFormatter()
        self.consensus_model = None
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self, level: str = 'INFO'):
        """Setup logging configuration."""
        log_level = getattr(logging, level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stderr)]
        )
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description="OptEnzML - Enzyme Optimal Temperature Prediction Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s --sequence "MKKLVLSLSLVLAFSSATAAF..." --ogt 95
  %(prog)s --fasta proteins.fasta --ogt 85 --output csv
  %(prog)s --sequence "MKK..." --ogt 75 --verbose --output json
  %(prog)s --directory /path/to/fastas --output-file results.csv
  %(prog)s --list-predictors
            """
        )
        
        # Input options
        input_group = parser.add_mutually_exclusive_group(required=False)
        input_group.add_argument(
            '--sequence', '-s',
            help='Protein sequence to analyze (single letter amino acid codes)'
        )
        input_group.add_argument(
            '--fasta', '-f',
            help='FASTA file containing protein sequences'
        )
        input_group.add_argument(
            '--directory', '-d',
            help='Directory containing FASTA files'
        )
        
        # Prediction parameters
        parser.add_argument(
            '--ogt',
            type=float,
            help='Optimal growth temperature of source organism (°C)'
        )
        
        # Predictor selection
        parser.add_argument(
            '--predictors',
            nargs='+',
            choices=['tomer', 'seq2topt', 'custom_rf', 'custom_svm', 'all'],
            default=['all'],
            help='Predictors to use (default: all available)'
        )
        
        # Consensus options
        parser.add_argument(
            '--consensus-method',
            choices=['auto', 'simple_average', 'weighted_average', 'ml_ensemble'],
            default='auto',
            help='Consensus method to use (default: auto)'
        )
        
        # Output options
        parser.add_argument(
            '--output', '-o',
            choices=['table', 'csv', 'json'],
            default='table',
            help='Output format (default: table)'
        )
        parser.add_argument(
            '--output-file',
            help='Output file path (default: stdout)'
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Show detailed results including individual predictor outputs'
        )
        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress progress messages'
        )
        
        # Tool configuration
        parser.add_argument(
            '--tomer-path',
            help='Path to TOMER executable'
        )
        parser.add_argument(
            '--seq2topt-path',
            help='Path to Seq2Topt executable'
        )
        parser.add_argument(
            '--consensus-model-path',
            help='Path to pre-trained consensus model'
        )
        
        # Information commands
        parser.add_argument(
            '--list-predictors',
            action='store_true',
            help='List available predictors and exit'
        )
        parser.add_argument(
            '--version',
            action='version',
            version='OptEnzML 1.0.0'
        )
        
        # Advanced options
        parser.add_argument(
            '--use-mock-predictors',
            action='store_true',
            help='Use mock predictors for testing (when external tools unavailable)'
        )
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Logging level (default: INFO)'
        )
        
        return parser
    
    def initialize_predictors(self, args) -> List:
        """Initialize predictor instances based on arguments."""
        predictors = []
        
        # Determine which predictors to use
        if 'all' in args.predictors:
            predictor_names = ['tomer', 'seq2topt', 'custom_rf', 'custom_svm']
        else:
            predictor_names = args.predictors
        
        # Initialize TOMER predictor
        if 'tomer' in predictor_names:
            if args.use_mock_predictors:
                predictors.append(MockTomerPredictor())
            else:
                predictors.append(TomerPredictor(args.tomer_path))
        
        # Initialize Seq2Topt predictor
        if 'seq2topt' in predictor_names:
            if args.use_mock_predictors:
                predictors.append(MockSeq2ToptPredictor())
            else:
                predictors.append(Seq2ToptPredictor(args.seq2topt_path))
        
        # Initialize custom predictors
        if 'custom_rf' in predictor_names:
            predictors.append(CustomRFPredictor())
        
        if 'custom_svm' in predictor_names:
            predictors.append(CustomSVMPredictor())
        
        return predictors
    
    def get_input_sequences(self, args) -> List[tuple]:
        """Get input sequences based on arguments."""
        sequences = []
        
        if args.sequence:
            # Single sequence
            try:
                clean_seq = self.data_loader.load_sequence_from_string(args.sequence)
                sequences.append(("command_line", clean_seq))
            except ValueError as e:
                raise ValueError(f"Invalid sequence: {e}")
        
        elif args.fasta:
            # FASTA file
            if not os.path.exists(args.fasta):
                raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
            
            fasta_sequences = self.data_loader.load_fasta_file(args.fasta)
            sequences.extend([(header, seq) for header, seq in fasta_sequences])
        
        elif args.directory:
            # Directory of FASTA files
            if not os.path.isdir(args.directory):
                raise NotADirectoryError(f"Directory not found: {args.directory}")
            
            dir_sequences = self.data_loader.load_sequences_from_directory(args.directory)
            sequences.extend([(f"{filename}:{header}", seq) for filename, header, seq in dir_sequences])
        
        else:
            raise ValueError("No input specified. Use --sequence, --fasta, or --directory")
        
        if not sequences:
            raise ValueError("No valid sequences found in input")
        
        return sequences
    
    def run_predictions(self, sequences: List[tuple], args) -> List:
        """Run predictions on sequences."""
        results = []
        total_sequences = len(sequences)
        
        for i, (header, sequence) in enumerate(sequences, 1):
            if not args.quiet:
                print(f"Processing sequence {i}/{total_sequences}: {header[:50]}...", file=sys.stderr)
            
            try:
                result = self.consensus_model.predict(
                    sequence=sequence,
                    ogt=args.ogt,
                    method=args.consensus_method
                )
                results.append(result)
            
            except Exception as e:
                logging.error(f"Error processing sequence {i}: {e}")
                # Create error result
                from ..consensus.consensus_model import ConsensusResult
                error_result = ConsensusResult(
                    error=str(e),
                    success=False
                )
                results.append(error_result)
        
        return results
    
    def format_and_output_results(self, results: List, args):
        """Format and output results."""
        try:
            if len(results) == 1:
                # Single result
                output = self.output_formatter.format_consensus_result(
                    results[0], args.output, args.verbose
                )
            else:
                # Multiple results
                output = self.output_formatter.format_multiple_results(
                    results, args.output, args.verbose
                )
            
            # Output to file or stdout
            if args.output_file:
                self.output_formatter.save_results_to_file(output, args.output_file)
                if not args.quiet:
                    print(f"Results saved to: {args.output_file}", file=sys.stderr)
            else:
                print(output)
        
        except Exception as e:
            logging.error(f"Error formatting output: {e}")
            raise
    
    def list_predictors(self, predictors: List):
        """List available predictors."""
        try:
            predictor_info = []
            for p in predictors:
                try:
                    info = p.get_info()
                    predictor_info.append(info)
                except Exception as e:
                    # Create basic info for failed predictors
                    predictor_info.append({
                        'name': getattr(p, 'name', 'Unknown'),
                        'version': getattr(p, 'version', 'Unknown'),
                        'is_available': getattr(p, 'is_available', False),
                        'error': str(e)
                    })
            
            output = self.output_formatter.format_predictor_info(predictor_info)
            print(output)
        except Exception as e:
            print(f"Error listing predictors: {e}", file=sys.stderr)
    
    def run(self, args=None):
        """Run the CLI application."""
        parser = self.create_parser()
        args = parser.parse_args(args)
        
        # Setup logging level
        self.setup_logging(args.log_level)
        
        try:
            # Initialize predictors
            predictors = self.initialize_predictors(args)
            
            if not predictors:
                print("Error: No predictors available", file=sys.stderr)
                return 1
            
            # Initialize consensus model
            self.consensus_model = ConsensusModel(
                predictors=predictors,
                consensus_model_path=args.consensus_model_path
            )
            
            # Handle list predictors command
            if args.list_predictors:
                self.list_predictors(predictors)
                return 0
            
            # Get input sequences
            sequences = self.get_input_sequences(args)
            
            if not args.quiet:
                available_predictors = self.consensus_model.get_available_predictors()
                print(f"Using {len(available_predictors)} predictors: {', '.join(available_predictors)}", file=sys.stderr)
                print(f"Processing {len(sequences)} sequence(s)...", file=sys.stderr)
            
            # Run predictions
            results = self.run_predictions(sequences, args)
            
            # Format and output results
            self.format_and_output_results(results, args)
            
            # Summary statistics
            if not args.quiet:
                successful = sum(1 for r in results if r.success)
                print(f"Completed: {successful}/{len(results)} successful predictions", file=sys.stderr)
            
            return 0
        
        except KeyboardInterrupt:
            print("\nInterrupted by user", file=sys.stderr)
            return 130
        
        except Exception as e:
            logging.error(f"Error: {e}")
            if args.log_level == 'DEBUG':
                import traceback
                traceback.print_exc()
            return 1


def main():
    """Main entry point for the CLI."""
    cli = OptEnzMLCLI()
    sys.exit(cli.run())


if __name__ == '__main__':
    main()
