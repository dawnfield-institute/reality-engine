"""
Quantify Emergent Physical Laws

Measure what laws emerge from the Reality Engine and compare to known physics.
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add parent to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.reality_engine import RealityEngine
from tools.law_quantifier import LawQuantifier
from tools.emergence_observer import EmergenceObserver

def quantify_emergent_laws(steps: int = 5000, size: tuple = (128, 32)):
    """Run simulation and quantify emergent physical laws."""
    
    print("=" * 70)
    print("LAW QUANTIFICATION - Dawn Field Theory")
    print("=" * 70)
    print()
    print(f"Measuring emergent laws over {steps} steps on {size[0]}×{size[1]} grid")
    print("No assumptions - discovering what physics actually emerges!")
    print()
    
    # Initialize engine
    print("Initializing reality engine...")
    engine = RealityEngine(size=size, dt=0.01)
    engine.initialize('big_bang')
    
    # Initialize analyzers
    quantifier = LawQuantifier()
    observer = EmergenceObserver()
    
    # Collect trajectory data
    trajectory = []
    
    print(f"Running {steps} steps and collecting field data...")
    print()
    
    for i in range(steps + 1):
        if i > 0:
            state_dict = engine.step()
        
        # Store full state periodically
        if i % 10 == 0:
            # Observe structures
            structures = observer.observe(engine.current_state)
            
            # Get field arrays
            A = engine.current_state.A.cpu().numpy() if torch.is_tensor(engine.current_state.A) else engine.current_state.A
            P = engine.current_state.P.cpu().numpy() if torch.is_tensor(engine.current_state.P) else engine.current_state.P
            M = engine.current_state.M.cpu().numpy() if torch.is_tensor(engine.current_state.M) else engine.current_state.M
            
            # Get QPL phase (for QBE constraint testing)
            qpl_phase = engine.qbe.compute_qpl(engine.time_elapsed) if hasattr(engine, 'qbe') else 0.0
            
            # Store trajectory point
            trajectory.append({
                'step': i,
                'A': A,
                'P': P,
                'M': M,
                'temperature': float(np.mean(np.abs(A - P))),
                'entropy': float(-np.sum(A * np.log(np.abs(A) + 1e-10))),
                'total_energy': float(np.sum((A - P)**2)),
                'qpl_phase': float(qpl_phase),
                'structures': structures
            })
            
            if i % 100 == 0:
                print(f"  Step {i:4d}: {len(structures)} structures detected")
    
    print()
    print("=" * 70)
    print("ANALYZING EMERGENT LAWS")
    print("=" * 70)
    
    # Measure conservation laws
    print("\n[*] 1. CONSERVATION LAWS")
    print("-" * 70)
    conservation_laws = quantifier.measure_conservation_laws(trajectory)
    
    if conservation_laws:
        for name, law in conservation_laws.items():
            print(f"\n  * {law.name}")
            print(f"    Equation: {law.equation}")
            print(f"    Confidence: {law.confidence:.3f}")
            print(f"    Parameters: {law.parameters}")
            if law.known_match:
                print(f"    [MATCH] Known law: {law.known_match}")
            if law.type == "quantum":
                print(f"    [QUANTUM] Quantum behavior detected!")
    else:
        print("  No conserved quantities detected")
    
    # Measure thermodynamic laws
    print("\n[*] 2. THERMODYNAMIC LAWS")
    print("-" * 70)
    thermo_laws = quantifier.measure_thermodynamic_laws(trajectory)
    
    if thermo_laws:
        for name, law in thermo_laws.items():
            print(f"\n  * {law.name}")
            print(f"    Equation: {law.equation}")
            print(f"    Confidence: {law.confidence:.3f}")
            print(f"    Parameters: {law.parameters}")
            if law.known_match:
                print(f"    [MATCH] Known law: {law.known_match}")
    else:
        print("  No thermodynamic laws detected")
    
    # Measure symmetries (on final state)
    print("\n[*] 3. SYMMETRIES")
    print("-" * 70)
    symmetries = quantifier.measure_symmetries(engine.current_state)
    
    if symmetries:
        for name, law in symmetries.items():
            print(f"\n  * {law.name}")
            print(f"    Equation: {law.equation}")
            print(f"    Confidence: {law.confidence:.3f}")
            if law.known_match:
                print(f"    [MATCH] Known symmetry: {law.known_match}")
    else:
        print("  No strong symmetries detected")
    
    # Measure force laws
    print("\n[*] 4. FORCE LAWS")
    print("-" * 70)
    print("  Analyzing structure interactions...")
    force_laws = quantifier.measure_force_laws(trajectory)
    
    if force_laws:
        for name, law in force_laws.items():
            print(f"\n  * {law.name}")
            print(f"    Equation: {law.equation}")
            print(f"    Parameters: {law.parameters}")
            print(f"    Confidence: {law.confidence:.3f}")
            if law.known_match:
                print(f"    [MATCH] Known law: {law.known_match}")
    else:
        print("  No force laws detected (structures may not interact enough)")
    
    # Compare to standard model
    print("\n" + "=" * 70)
    print("COMPARISON TO STANDARD PHYSICS")
    print("=" * 70)
    
    all_laws = {}
    all_laws.update(conservation_laws)
    all_laws.update(thermo_laws)
    all_laws.update(symmetries)
    all_laws.update(force_laws)
    
    comparison = quantifier.compare_to_standard_model(all_laws)
    
    print(f"\n{comparison['summary']}")
    
    if comparison['matched_laws']:
        print("\n[MATCHED] Known Laws:")
        for match in comparison['matched_laws']:
            print(f"  * {match['known']}: {match['discovered']}")
            print(f"    Confidence: {match['confidence']:.3f}, Deviation: {match['deviation']:.4f}")
    
    if comparison['novel_laws']:
        print("\n[NOVEL] New Laws Discovered:")
        for novel in comparison['novel_laws']:
            print(f"  * {novel['name']}: {novel['equation']}")
            print(f"    Parameters: {novel['parameters']}")
            print(f"    Confidence: {novel['confidence']:.3f}")
    
    if comparison['missing_laws']:
        print("\n[MISSING] Expected Laws Not Yet Detected:")
        for missing in comparison['missing_laws']:
            print(f"  * {missing}")
    
    # Save results
    output_dir = Path("output/law_quantification")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"laws_{timestamp}.json"
    
    # Convert laws to dict for JSON serialization
    def law_to_dict(law):
        return {
            'name': law.name,
            'type': law.type,
            'equation': law.equation,
            'parameters': law.parameters,
            'confidence': float(law.confidence),
            'known_match': law.known_match,
            'deviation': float(law.deviation),
            'observations': law.observations
        }
    
    results = {
        'parameters': {
            'steps': steps,
            'size': list(size),
            'dt': 0.01
        },
        'conservation_laws': {k: law_to_dict(v) for k, v in conservation_laws.items()},
        'thermodynamic_laws': {k: law_to_dict(v) for k, v in thermo_laws.items()},
        'symmetries': {k: law_to_dict(v) for k, v in symmetries.items()},
        'force_laws': {k: law_to_dict(v) for k, v in force_laws.items()},
        'comparison': comparison,
        'total_laws_discovered': len(all_laws)
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("=" * 70)
    print("[COMPLETE] LAW QUANTIFICATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_file}")
    print()
    print("Key Findings:")
    print(f"  • Total laws discovered: {len(all_laws)}")
    print(f"  • Matched to known physics: {len(comparison['matched_laws'])}")
    print(f"  • Novel phenomena: {len(comparison['novel_laws'])}")
    print()
    print("Everything emerged from: B(x,t) = ∇²(E-I) + λM∇²M - α||E-I||²")
    
    return results

if __name__ == "__main__":
    import torch
    
    print()
    results = quantify_emergent_laws(steps=1000, size=(128, 32))
    print()
