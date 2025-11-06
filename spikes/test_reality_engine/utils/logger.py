"""
Reality Engine Logger

Comprehensive logging system that captures:
- Field statistics (E, I, M values and ranges)
- Conservation metrics (PAC total, drift)
- Emergence events (collapses, quantum, particles)
- Performance metrics (GPU usage, step time)
- Warnings and errors (NaN, instability)

All logs saved to logs/ directory with timestamps.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json


class RealityLogger:
    """
    Comprehensive logger for Reality Engine simulations.
    
    Creates both human-readable logs and machine-readable JSON logs.
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "reality_engine", output_dir: str = "output"):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name prefix for log files
            output_dir: Base directory for output files (timestamped folder created inside)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        self.json_file = self.log_dir / f"{experiment_name}_{timestamp}.json"
        
        # Create timestamped output directory
        self.output_dir = Path(output_dir) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self.logger = logging.getLogger(f"RealityEngine_{timestamp}")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler (detailed) - UTF-8 encoding for emojis
        fh = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # Console handler (less verbose) - UTF-8 encoding
        import sys
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # JSON log for machine-readable data
        self.json_data = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'output_dir': str(self.output_dir),
            'steps': [],
            'events': [],
            'warnings': []
        }
        
        self.logger.info("="*70)
        self.logger.info(f"Reality Engine Log: {self.log_file.name}")
        self.logger.info(f"Output Directory: {self.output_dir}")
        self.logger.info("="*70)
    
    def log_initialization(self, config: Dict[str, Any]):
        """Log initialization parameters."""
        self.logger.info("\n" + "="*70)
        self.logger.info("INITIALIZATION")
        self.logger.info("="*70)
        
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        
        self.json_data['config'] = config
    
    def log_step(self, step: int, stats: Dict[str, Any], detailed: bool = False):
        """
        Log a simulation step.
        
        Args:
            step: Step number
            stats: Statistics dictionary
            detailed: If True, log full field statistics
        """
        # Check for NaN/Inf
        has_nan = any(
            str(v) in ['nan', 'inf', '-inf'] 
            for v in stats.values() 
            if isinstance(v, (int, float))
        )
        
        if has_nan:
            self.logger.error(f"⚠️  STEP {step}: NaN/Inf DETECTED!")
            self.logger.error(f"  Stats: {stats}")
            self.json_data['warnings'].append({
                'step': step,
                'type': 'NaN',
                'stats': stats
            })
        
        # Log basic stats every step if detailed, or on warnings
        if detailed or has_nan or step % 100 == 0:
            msg = (
                f"Step {step:5d} | "
                f"E={stats.get('mean_energy', 0):.6f} "
                f"I={stats.get('mean_info', 0):.6f} "
                f"M={stats.get('mean_memory', 0):.6f} | "
                f"PAC={stats.get('total_pac', 0):.2f} | "
                f"H={stats.get('herniations', 0):3d}"
            )
            
            if has_nan:
                self.logger.error(msg)
            else:
                self.logger.debug(msg)
        
        # Store in JSON
        if step % 10 == 0 or has_nan:  # Store every 10th step + warnings
            self.json_data['steps'].append({
                'step': step,
                **stats
            })
    
    def log_event(self, event_type: str, step: int, details: Dict[str, Any]):
        """
        Log an emergence event.
        
        Args:
            event_type: Type of event ('quantum', 'particle', 'gravity', etc.)
            step: Step when event occurred
            details: Event details
        """
        self.logger.info(f"🔬 {event_type.upper()} at step {step}")
        for key, value in details.items():
            self.logger.info(f"     {key}: {value}")
        
        self.json_data['events'].append({
            'type': event_type,
            'step': step,
            'details': details
        })
    
    def log_warning(self, message: str, details: Optional[Dict] = None):
        """Log a warning."""
        self.logger.warning(f"⚠️  {message}")
        if details:
            for key, value in details.items():
                self.logger.warning(f"    {key}: {value}")
        
        self.json_data['warnings'].append({
            'message': message,
            'details': details
        })
    
    def log_phase(self, phase_name: str, description: str = ""):
        """Log a new simulation phase."""
        self.logger.info("\n" + "="*70)
        self.logger.info(f"{phase_name}")
        if description:
            self.logger.info(f"{description}")
        self.logger.info("="*70)
    
    def log_validation(self, results: Dict[str, Any]):
        """Log validation results."""
        self.logger.info("\n" + "="*70)
        self.logger.info("VALIDATION RESULTS")
        self.logger.info("="*70)
        
        for key, value in results.items():
            if isinstance(value, bool):
                symbol = "[OK]" if value else "[NO]"
                self.logger.info(f"  {symbol} {key}: {value}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        self.json_data['validation'] = results
    
    def log_field_diagnostics(self, field_name: str, field_stats: Dict[str, Any]):
        """Log detailed field diagnostics."""
        self.logger.debug(f"\n{field_name} Field Diagnostics:")
        self.logger.debug(f"  Min: {field_stats.get('min', 'N/A')}")
        self.logger.debug(f"  Max: {field_stats.get('max', 'N/A')}")
        self.logger.debug(f"  Mean: {field_stats.get('mean', 'N/A')}")
        self.logger.debug(f"  Std: {field_stats.get('std', 'N/A')}")
        self.logger.debug(f"  Sum: {field_stats.get('sum', 'N/A')}")
    
    def save(self):
        """Save JSON log to file."""
        with open(self.json_file, 'w') as f:
            json.dump(self.json_data, f, indent=2)
        
        # Write run_info.json to output directory
        run_info = {
            'timestamp': self.timestamp,
            'config': self.json_data.get('config', {}),
            'total_steps': len(self.json_data['steps']),
            'total_events': len(self.json_data['events']),
            'total_warnings': len(self.json_data['warnings']),
            'log_file': str(self.log_file),
            'json_file': str(self.json_file),
            'output_dir': str(self.output_dir)
        }
        
        run_info_file = self.output_dir / 'run_info.json'
        with open(run_info_file, 'w') as f:
            json.dump(run_info, f, indent=2)
        
        self.logger.info(f"\n[SAVED] Logs saved:")
        self.logger.info(f"  Text: {self.log_file}")
        self.logger.info(f"  JSON: {self.json_file}")
        self.logger.info(f"  Run Info: {run_info_file}")
    
    def close(self):
        """Close logger and save data."""
        self.save()
        
        # Close handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def get_field_stats(field) -> Dict[str, float]:
    """
    Extract statistics from a field tensor.
    
    Args:
        field: PyTorch tensor
        
    Returns:
        Dictionary of statistics
    """
    import torch
    
    # Handle NaN/Inf
    has_nan = torch.isnan(field).any().item()
    has_inf = torch.isinf(field).any().item()
    
    if has_nan or has_inf:
        return {
            'min': float('nan'),
            'max': float('nan'),
            'mean': float('nan'),
            'std': float('nan'),
            'sum': float('nan'),
            'has_nan': has_nan,
            'has_inf': has_inf
        }
    
    return {
        'min': field.min().item(),
        'max': field.max().item(),
        'mean': field.mean().item(),
        'std': field.std().item(),
        'sum': field.sum().item(),
        'has_nan': False,
        'has_inf': False
    }


if __name__ == '__main__':
    # Test logger
    logger = RealityLogger(experiment_name="test")
    
    logger.log_initialization({
        'size': 64,
        'dt': 0.0001,
        'device': 'cuda'
    })
    
    logger.log_phase("BIG BANG", "Initiating field")
    
    for i in range(5):
        logger.log_step(i, {
            'mean_energy': 0.5 - i*0.01,
            'mean_info': 0.3 + i*0.02,
            'mean_memory': i*0.001,
            'total_pac': 100.0,
            'herniations': 10 - i
        })
    
    logger.log_event('quantum', 2, {
        'coherence': 0.95,
        'entanglement': 0.12
    })
    
    logger.log_warning("Test warning", {'field': 'energy', 'value': 'high'})
    
    logger.log_validation({
        'pac_conserved': True,
        'quantum_emerged': True,
        'particles_emerged': False
    })
    
    logger.close()
    print(f"\n✓ Test complete! Check logs/ directory")
