"""
Reality Engine - Example Run

Demonstrates how to use the Reality Engine to simulate
the emergence of physics from pure field dynamics.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from engine import RealityEngine
from visualize_fields import visualize_all_fields
from utils.logger import RealityLogger


def main():
    """Run a simple Reality Engine simulation."""
    
    # Create logger (this will create timestamped output directory)
    logger = RealityLogger(experiment_name="reality_engine")
    
    logger.log_phase("REALITY ENGINE", "Pure Field Evolution Experiment - Based on Dawn Field Theory")
    
    # Configuration
    config = {
        'shape': (64, 64, 64),
        'big_bang_perturbation': 1e-10,
        'evolution_steps': 10000,
        'check_interval': 500
    }
    
    logger.log_initialization(config)
    
    # Create engine with smaller field for faster demo
    logger.log_phase("INITIALIZATION", "Creating field engine")
    engine = RealityEngine(shape=config['shape'])
    
    # Trigger Big Bang
    logger.log_phase("BIG BANG", "Initiating herniation cascade")
    spacetime = engine.big_bang(seed_perturbation=config['big_bang_perturbation'])
    
    logger.log_event('big_bang', 0, {
        'total_herniations': spacetime.get('total_herniations', 0),
        'cascade_depth': spacetime.get('cascade_depth', 0),
        'pac_total': spacetime.get('pac_total', 0)
    })
    
    # Evolve for 10000 steps to see emergence
    logger.log_phase("EVOLUTION", f"Evolving for {config['evolution_steps']} steps")
    
    try:
        report = engine.evolve(
            steps=config['evolution_steps'], 
            check_interval=config['check_interval'], 
            verbose=True
            # TODO: Add logger parameter to engine.evolve()
        )
        
        # Log steps manually for now
        for i, step_data in enumerate(engine.history.get('pac', [])):
            if i % 100 == 0:
                logger.log_step(i, {
                    'mean_energy': engine.history['energy'][i] if i < len(engine.history['energy']) else 0,
                    'mean_info': engine.history['information'][i] if i < len(engine.history['information']) else 0,
                    'mean_memory': engine.history['memory'][i] if i < len(engine.history['memory']) else 0,
                    'total_pac': step_data,
                    'herniations': engine.history['herniations'][i] if i < len(engine.history['herniations']) else 0
                })
    except Exception as e:
        logger.log_warning(f"Evolution failed: {e}")
        raise
    
    # Print detailed results
    logger.log_phase("VALIDATION", "Checking emergent phenomena")
    
    validation = report['validation']
    logger.log_validation({
        'PAC Conservation': validation['pac_conservation'],
        'Quantum Emerged': validation['quantum_emerged'],
        'Particles Emerged': validation['particles_emerged'],
        'Gravity Emerged': validation['gravity_emerged']
    })
    
    # Generate CMB-like visualizations
    logger.log_phase("VISUALIZATION", "Creating field images")
    visualize_all_fields(engine.field, output_dir=str(logger.output_dir))
    
    # Save results to timestamped output directory
    output_file = logger.output_dir / "reality_engine_results.pkl"
    engine.save_state(str(output_file))
    
    logger.log_phase("COMPLETE", f"Results saved to {logger.output_dir}/")
    logger.close()
    
    return report


if __name__ == "__main__":
    report = main()
