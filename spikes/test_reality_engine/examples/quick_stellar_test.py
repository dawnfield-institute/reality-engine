"""
Quick Stellar Test - Test compression heating

Runs a shorter evolution (10k steps) to quickly see if compression 
heating creates stars with fusion vs just cold black holes.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from datetime import datetime

from core.dawn_field import DawnField
from emergence.particle_analyzer import ParticleAnalyzer
from emergence.stellar_analyzer import StellarAnalyzer


def quick_stellar_test():
    """Quick test to see if compression heating creates stars"""
    
    print("="*70)
    print("QUICK STELLAR TEST - Compression Heating")
    print("="*70)
    print("Testing if gravitational compression creates stars with fusion")
    print("="*70)
    
    # Shorter run for quick testing
    total_steps = 10000
    check_interval = 2000
    
    print(f"\n[1/3] Initializing (64³ universe)...")
    reality = DawnField(shape=(64, 64, 64), dt=0.0001, device='cuda')
    
    stellar_analyzer = StellarAnalyzer(mass_threshold=500.0)
    
    print(f"\n[2/3] Evolving {total_steps:,} steps...")
    
    temp_history = []
    stellar_history = []
    
    for step in range(total_steps):
        reality.evolve_step()
        
        if step > 0 and step % check_interval == 0:
            # Check fields
            E_np = reality.E.cpu().numpy()
            I_np = reality.I.cpu().numpy()
            M_np = reality.M.cpu().numpy()
            
            # Detect stellar structures
            structures = stellar_analyzer.detect_structures(E_np, I_np, M_np)
            
            # Check for fusion
            fusion_events = stellar_analyzer.detect_fusion_events(E_np, M_np, step)
            
            print(f"\n{'='*60}")
            print(f"Step {step:,}")
            print(f"{'='*60}")
            
            if structures:
                print(f"Stellar structures: {len(structures)}")
                for s in structures:
                    print(f"  {s.structure_type:20s}: "
                          f"M={s.total_mass:7.1f}, "
                          f"T={s.temperature:6.2f}, "
                          f"ρ={s.core_density:5.2f}")
                    
                    if s.is_fusion_active():
                        print(f"    >>> FUSION ACTIVE! <<<")
                
                # Track temperature evolution
                avg_temp = np.mean([s.temperature for s in structures])
                max_temp = max([s.temperature for s in structures])
                
                temp_history.append({
                    'step': step,
                    'avg_temp': avg_temp,
                    'max_temp': max_temp,
                    'structures': len(structures)
                })
                
                # Track structure types
                types = {}
                for s in structures:
                    types[s.structure_type] = types.get(s.structure_type, 0) + 1
                
                stellar_history.append({
                    'step': step,
                    'types': types,
                    'fusion_count': len(fusion_events)
                })
                
                if fusion_events:
                    print(f"\n!!! {len(fusion_events)} FUSION EVENTS !!!")
            else:
                print("No stellar structures yet")
        
        if step % 500 == 0:
            print(f"Progress: {100*step/total_steps:.1f}%", end='\r')
    
    print(f"\n\n[3/3] Final Analysis")
    print("="*70)
    
    E_np = reality.E.cpu().numpy()
    I_np = reality.I.cpu().numpy()
    M_np = reality.M.cpu().numpy()
    
    structures = stellar_analyzer.detect_structures(E_np, I_np, M_np)
    
    stellar_analyzer.print_summary()
    
    # Check if we got stars vs black holes
    if structures:
        has_stars = any('star' in s.structure_type for s in structures)
        has_fusion = any(s.is_fusion_active() for s in structures)
        max_temp = max([s.temperature for s in structures])
        
        print("\n" + "="*70)
        print("COMPRESSION HEATING RESULTS")
        print("="*70)
        
        if has_fusion:
            print("✓ SUCCESS: Fusion detected!")
            print(f"  Created {sum(1 for s in structures if s.is_fusion_active())} fusion regions")
        else:
            print("✗ No fusion yet")
        
        if has_stars:
            print(f"✓ SUCCESS: Stars formed!")
            star_count = sum(1 for s in structures if 'star' in s.structure_type)
            print(f"  Created {star_count} stellar objects")
        else:
            print("✗ Only black holes formed (cold collapse)")
        
        print(f"\nTemperature range: {max_temp:.3f}")
        print(f"  (Need T > 10 for fusion)")
        
        if max_temp > 0.01:
            print("\n✓ Compression heating is working!")
            print("  Temperature increased from 0 → {:.3f}".format(max_temp))
        else:
            print("\n⚠ Temperature still near zero")
            print("  May need stronger compression heating")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    quick_stellar_test()
