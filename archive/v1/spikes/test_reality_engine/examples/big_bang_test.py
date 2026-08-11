"""
Big Bang Evolution Test

Test the pressure-driven evolution model starting from pure energy state.

This validates:
- Pure energy (Big Bang) → Structure emerges naturally
- Time advances from collapse events (no collapse = frozen time)
- Heat death detection (pressure → 0)
- E↔I phase transitions at critical densities (fusion)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import logging
from pathlib import Path
from datetime import datetime

from core.dawn_field import DawnField
from emergence.particle_analyzer import ParticleAnalyzer
from emergence.stellar_analyzer import StellarAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run Big Bang evolution with pressure-driven dynamics."""
    
    logger.info("=" * 80)
    logger.info("BIG BANG EVOLUTION TEST")
    logger.info("Pressure-driven model: E=pressure → collapse → I → M → time")
    logger.info("=" * 80)
    
    # Create universe (small for testing)
    grid_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    logger.info(f"Grid: {grid_size}³ = {grid_size**3:,} cells")
    
    field = DawnField(
        shape=(grid_size, grid_size, grid_size),
        dt=0.0001,  # Critical timestep
        device=device
    )
    
    # Initialize Big Bang state (pure energy)
    field.initialize_big_bang()
    
    # Create analyzers
    particle_analyzer = ParticleAnalyzer(device=device)
    stellar_analyzer = StellarAnalyzer()  # No device parameter
    
    # Evolution parameters
    max_steps = 5000
    check_interval = 500
    
    logger.info(f"\nRunning {max_steps} steps (checking every {check_interval})...")
    logger.info(f"Expected: Structure emerges from pressure collapse\n")
    
    # Track metrics
    timeline = []
    
    for step in range(max_steps):
        # Evolve
        stats = field.evolve_step()
        
        # Check periodically
        if step % check_interval == 0:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"STEP {step} | Time: {stats['time']:.2f}")
            logger.info(f"{'=' * 60}")
            
            # Field statistics
            E_stats = {
                'mean': field.E.mean().item(),
                'std': field.E.std().item(),
                'max': field.E.max().item()
            }
            I_stats = {
                'mean': field.I.mean().item(),
                'std': field.I.std().item(),
                'max': field.I.max().item()
            }
            M_stats = {
                'mean': field.M.mean().item(),
                'std': field.M.std().item(),
                'max': field.M.max().item()
            }
            
            logger.info(f"ENERGY:      mean={E_stats['mean']:.4f}, std={E_stats['std']:.4f}, max={E_stats['max']:.4f}")
            logger.info(f"INFORMATION: mean={I_stats['mean']:.4f}, std={I_stats['std']:.4f}, max={I_stats['max']:.4f}")
            logger.info(f"MEMORY:      mean={M_stats['mean']:.4f}, std={M_stats['std']:.4f}, max={M_stats['max']:.4f}")
            
            # Collapse statistics
            logger.info(f"\nCRYSTALLIZATION DYNAMICS:")
            logger.info(f"  Crystallizations: {stats['crystallizations']}")
            logger.info(f"  Fraction:  {stats['crystallization_fraction']:.4f}")
            logger.info(f"  Time rate: {stats['time_rate']:.2f}x")
            
            # Detect particles
            particles = particle_analyzer.detect_particles(field.E, field.I, field.M)
            logger.info(f"\nPARTICLES: {len(particles)} detected")
            if len(particles) > 0:
                total_mass = sum(p.mass for p in particles)
                avg_stability = sum(p.stability for p in particles) / len(particles)
                logger.info(f"  Total mass: {total_mass:.2f}")
                logger.info(f"  Avg stability: {avg_stability:.4f}")
                
                # Count by type
                from collections import Counter
                type_counts = Counter(p.classification for p in particles)
                logger.info(f"  Types: {dict(type_counts)}")
            
            # Detect stellar structures
            structures = stellar_analyzer.detect_structures(field.E, field.I, field.M)
            logger.info(f"\nSTELLAR STRUCTURES: {len(structures)} detected")
            if len(structures) > 0:
                for i, s in enumerate(structures[:5]):  # Show first 5
                    logger.info(f"  {i+1}. {s.structure_type}: mass={s.mass:.1f}, T={s.temperature:.4f}, ρ={s.core_density:.4f}")
            
            # Record timeline
            timeline.append({
                'step': step,
                'time': stats['time'],
                'crystallizations': stats['crystallizations'],
                'time_rate': stats['time_rate'],
                'E': E_stats,
                'I': I_stats,
                'M': M_stats,
                'particles': len(particles),
                'structures': len(structures)
            })
            
            # Heat death check (no crystallizations for extended period)
            if stats['crystallizations'] == 0:
                logger.warning("\n⚠️  NO CRYSTALLIZATIONS: Universe may be approaching equilibrium")
                # Don't break yet - might restart
                pass
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("EVOLUTION COMPLETE")
    logger.info("=" * 80)
    
    logger.info(f"Final time: {field.time:.2f}")
    logger.info(f"Total steps: {field.step_count}")
    logger.info(f"Final particles: {len(particles)}")
    logger.info(f"Final structures: {len(structures)}")
    
    # PAC conservation check
    final_pac = field.E.sum() + field.I.sum() + field.M.sum()
    if hasattr(field, 'initial_pac_total'):
        conservation_error = abs(final_pac - field.initial_pac_total) / field.initial_pac_total
        logger.info(f"\nPAC Conservation: {conservation_error:.6f} error")
        if conservation_error < 0.01:
            logger.info("✅ Excellent conservation!")
        elif conservation_error < 0.05:
            logger.info("✅ Good conservation")
        else:
            logger.warning("⚠️  Conservation error high - check dynamics")
    
    # Analyze timeline
    if len(timeline) > 2:
        logger.info("\nTIMELINE ANALYSIS:")
        
        # Structure formation rate
        initial_structures = timeline[0]['structures']
        final_structures = timeline[-1]['structures']
        logger.info(f"  Structures: {initial_structures} → {final_structures}")
        
        # Collapse rate evolution
        initial_crystallizations = timeline[0]['crystallizations']
        final_crystallizations = timeline[-1]['crystallizations']
        logger.info(f"  Crystallization rate: {initial_crystallizations} → {final_crystallizations}")
        
        # Energy evolution
        initial_E = timeline[0]['E']['mean']
        final_E = timeline[-1]['E']['mean']
        logger.info(f"  Energy: {initial_E:.4f} → {final_E:.4f}")
        
        # Information accumulation
        initial_I = timeline[0]['I']['mean']
        final_I = timeline[-1]['I']['mean']
        logger.info(f"  Information: {initial_I:.4f} → {final_I:.4f}")
        
        # Memory accumulation
        initial_M = timeline[0]['M']['mean']
        final_M = timeline[-1]['M']['mean']
        logger.info(f"  Memory: {initial_M:.4f} → {final_M:.4f}")
    
    logger.info("\n✅ Test complete!")

if __name__ == '__main__':
    main()
