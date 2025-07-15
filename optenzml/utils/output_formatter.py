"""
Output formatting utilities for OptEnzML

This module provides utilities for formatting prediction results in various formats
including table, CSV, and JSON.
"""

import json
import csv
import io
from typing import List, Dict, Any, Optional
from ..consensus.consensus_model import ConsensusResult
from ..predictors.base_predictor import PredictionResult


class OutputFormatter:
    """
    Utility class for formatting OptEnzML prediction results.
    """
    
    def __init__(self):
        self.supported_formats = ['table', 'csv', 'json']
    
    def format_consensus_result(self, result: ConsensusResult, 
                              format_type: str = 'table',
                              verbose: bool = False) -> str:
        """
        Format a single consensus result.
        
        Args:
            result: ConsensusResult to format
            format_type: Output format ('table', 'csv', 'json')
            verbose: Whether to include detailed information
            
        Returns:
            Formatted result string
        """
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}. Supported: {self.supported_formats}")
        
        if format_type == 'json':
            return self._format_consensus_json(result, verbose)
        elif format_type == 'csv':
            return self._format_consensus_csv(result, verbose)
        else:  # table
            return self._format_consensus_table(result, verbose)
    
    def format_multiple_results(self, results: List[ConsensusResult],
                              format_type: str = 'table',
                              verbose: bool = False) -> str:
        """
        Format multiple consensus results.
        
        Args:
            results: List of ConsensusResult objects
            format_type: Output format ('table', 'csv', 'json')
            verbose: Whether to include detailed information
            
        Returns:
            Formatted results string
        """
        if format_type == 'json':
            return self._format_multiple_json(results, verbose)
        elif format_type == 'csv':
            return self._format_multiple_csv(results, verbose)
        else:  # table
            return self._format_multiple_table(results, verbose)
    
    def _format_consensus_json(self, result: ConsensusResult, verbose: bool) -> str:
        """Format consensus result as JSON."""
        output_data = {
            'success': result.success,
            'consensus_prediction': {
                'temperature': result.consensus_temp,
                'confidence': result.confidence,
                'method': result.consensus_method,
                'execution_time': result.execution_time
            }
        }
        
        if result.error:
            output_data['error'] = result.error
        
        if verbose and result.individual_predictions:
            output_data['individual_predictions'] = []
            for pred in result.individual_predictions:
                pred_data = {
                    'model': pred.model_name,
                    'success': pred.success,
                    'temperature': pred.predicted_temp,
                    'confidence': pred.confidence,
                    'execution_time': pred.execution_time
                }
                if pred.error:
                    pred_data['error'] = pred.error
                if pred.metadata:
                    pred_data['metadata'] = pred.metadata
                
                output_data['individual_predictions'].append(pred_data)
        
        if result.metadata:
            output_data['metadata'] = result.metadata
        
        return json.dumps(output_data, indent=2)
    
    def _format_consensus_csv(self, result: ConsensusResult, verbose: bool) -> str:
        """Format consensus result as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        if verbose:
            writer.writerow([
                'Type', 'Model', 'Success', 'Temperature', 'Confidence', 
                'Method', 'Execution_Time', 'Error'
            ])
            
            # Consensus row
            writer.writerow([
                'Consensus', result.consensus_method, result.success,
                result.consensus_temp, result.confidence, result.consensus_method,
                result.execution_time, result.error or ''
            ])
            
            # Individual predictions
            if result.individual_predictions:
                for pred in result.individual_predictions:
                    writer.writerow([
                        'Individual', pred.model_name, pred.success,
                        pred.predicted_temp, pred.confidence, '',
                        pred.execution_time, pred.error or ''
                    ])
        else:
            writer.writerow(['Temperature', 'Confidence', 'Method', 'Success'])
            writer.writerow([
                result.consensus_temp, result.confidence, 
                result.consensus_method, result.success
            ])
        
        return output.getvalue()
    
    def _format_consensus_table(self, result: ConsensusResult, verbose: bool) -> str:
        """Format consensus result as readable table."""
        lines = []
        
        lines.append("=" * 60)
        lines.append("OPTENZML PREDICTION RESULT")
        lines.append("=" * 60)
        
        if result.success:
            lines.append(f"Consensus Temperature: {result.consensus_temp}°C")
            lines.append(f"Confidence Score:     {result.confidence:.3f}")
            lines.append(f"Consensus Method:     {result.consensus_method}")
            lines.append(f"Execution Time:       {result.execution_time:.3f}s")
        else:
            lines.append(f"Prediction Failed: {result.error}")
        
        if verbose and result.individual_predictions:
            lines.append("")
            lines.append("Individual Predictor Results:")
            lines.append("-" * 60)
            lines.append(f"{'Model':<15} {'Success':<8} {'Temp (°C)':<10} {'Confidence':<11} {'Time (s)':<8}")
            lines.append("-" * 60)
            
            for pred in result.individual_predictions:
                success_str = "✓" if pred.success else "✗"
                temp_str = f"{pred.predicted_temp:.1f}" if pred.predicted_temp else "N/A"
                conf_str = f"{pred.confidence:.3f}" if pred.confidence else "N/A"
                time_str = f"{pred.execution_time:.3f}" if pred.execution_time else "N/A"
                
                lines.append(f"{pred.model_name:<15} {success_str:<8} {temp_str:<10} {conf_str:<11} {time_str:<8}")
                
                if pred.error and not pred.success:
                    lines.append(f"  Error: {pred.error}")
        
        if result.metadata:
            lines.append("")
            lines.append("Metadata:")
            for key, value in result.metadata.items():
                lines.append(f"  {key}: {value}")
        
        return '\n'.join(lines)
    
    def _format_multiple_json(self, results: List[ConsensusResult], verbose: bool) -> str:
        """Format multiple results as JSON."""
        output_data = {
            'total_predictions': len(results),
            'successful_predictions': sum(1 for r in results if r.success),
            'results': []
        }
        
        for i, result in enumerate(results):
            result_data = json.loads(self._format_consensus_json(result, verbose))
            result_data['sequence_index'] = i
            output_data['results'].append(result_data)
        
        return json.dumps(output_data, indent=2)
    
    def _format_multiple_csv(self, results: List[ConsensusResult], verbose: bool) -> str:
        """Format multiple results as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        
        if verbose:
            # Detailed format with individual predictions
            writer.writerow([
                'Sequence_Index', 'Type', 'Model', 'Success', 'Temperature', 
                'Confidence', 'Method', 'Execution_Time', 'Error'
            ])
            
            for i, result in enumerate(results):
                # Consensus row
                writer.writerow([
                    i, 'Consensus', result.consensus_method, result.success,
                    result.consensus_temp, result.confidence, result.consensus_method,
                    result.execution_time, result.error or ''
                ])
                
                # Individual predictions
                if result.individual_predictions:
                    for pred in result.individual_predictions:
                        writer.writerow([
                            i, 'Individual', pred.model_name, pred.success,
                            pred.predicted_temp, pred.confidence, '',
                            pred.execution_time, pred.error or ''
                        ])
        else:
            # Summary format
            writer.writerow([
                'Sequence_Index', 'Temperature', 'Confidence', 'Method', 'Success'
            ])
            
            for i, result in enumerate(results):
                writer.writerow([
                    i, result.consensus_temp, result.confidence,
                    result.consensus_method, result.success
                ])
        
        return output.getvalue()
    
    def _format_multiple_table(self, results: List[ConsensusResult], verbose: bool) -> str:
        """Format multiple results as readable table."""
        lines = []
        
        lines.append("=" * 80)
        lines.append("OPTENZML BATCH PREDICTION RESULTS")
        lines.append("=" * 80)
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        lines.append(f"Total Predictions:      {len(results)}")
        lines.append(f"Successful Predictions: {len(successful)}")
        lines.append(f"Failed Predictions:     {len(failed)}")
        
        if successful:
            temps = [r.consensus_temp for r in successful if r.consensus_temp is not None]
            if temps:
                lines.append(f"Temperature Range:      {min(temps):.1f}°C - {max(temps):.1f}°C")
                lines.append(f"Average Temperature:    {sum(temps)/len(temps):.1f}°C")
        
        lines.append("")
        lines.append("Summary Results:")
        lines.append("-" * 80)
        lines.append(f"{'Index':<6} {'Success':<8} {'Temp (°C)':<10} {'Confidence':<11} {'Method':<15} {'Time (s)':<8}")
        lines.append("-" * 80)
        
        for i, result in enumerate(results):
            success_str = "✓" if result.success else "✗"
            temp_str = f"{result.consensus_temp:.1f}" if result.consensus_temp else "N/A"
            conf_str = f"{result.confidence:.3f}" if result.confidence else "N/A"
            method_str = result.consensus_method[:14] if result.consensus_method else "N/A"
            time_str = f"{result.execution_time:.3f}" if result.execution_time else "N/A"
            
            lines.append(f"{i:<6} {success_str:<8} {temp_str:<10} {conf_str:<11} {method_str:<15} {time_str:<8}")
        
        if verbose:
            lines.append("")
            lines.append("Detailed Results:")
            lines.append("=" * 80)
            
            for i, result in enumerate(results):
                lines.append(f"\nSequence {i+1}:")
                lines.append(self._format_consensus_table(result, True))
        
        return '\n'.join(lines)
    
    def format_predictor_info(self, predictors_info: List[Dict[str, Any]], 
                            format_type: str = 'table') -> str:
        """
        Format predictor information.
        
        Args:
            predictors_info: List of predictor info dictionaries
            format_type: Output format
            
        Returns:
            Formatted predictor information
        """
        if format_type == 'json':
            return json.dumps(predictors_info, indent=2)
        elif format_type == 'csv':
            output = io.StringIO()
            writer = csv.writer(output)
            
            if predictors_info:
                # Header
                keys = predictors_info[0].keys()
                writer.writerow(keys)
                
                # Data
                for info in predictors_info:
                    writer.writerow([info.get(key, '') for key in keys])
            
            return output.getvalue()
        else:  # table
            lines = []
            lines.append("Available Predictors:")
            lines.append("-" * 60)
            lines.append(f"{'Name':<20} {'Version':<10} {'Available':<10} {'Type':<15}")
            lines.append("-" * 60)
            
            for info in predictors_info:
                name = info.get('name', 'Unknown')[:19]
                version = info.get('version', 'N/A')[:9]
                available = "✓" if info.get('available', False) else "✗"
                pred_type = info.get('type', 'Unknown')[:14]
                
                lines.append(f"{name:<20} {version:<10} {available:<10} {pred_type:<15}")
            
            return '\n'.join(lines)
    
    def save_results_to_file(self, content: str, filepath: str) -> None:
        """
        Save formatted results to a file.
        
        Args:
            content: Formatted content to save
            filepath: Output file path
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            raise IOError(f"Error saving results to {filepath}: {e}")
