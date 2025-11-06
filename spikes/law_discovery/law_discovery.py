"""
Automated Law Discovery from Reality Engine Simulation

Analyzes simulation history to discover:
- Conservation laws (energy, matter)
- Thermodynamic laws (2nd law, Landauer principle)
- Emergent constants (c_effective, cooling rates, coupling constants)
- Phase transitions and critical points
- Information-thermodynamic correlations

This demonstrates that fundamental physics emerges from pure dynamics!
"""
import sys
from pathlib import Path
import json

# Add parent directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.reality_engine import RealityEngine

def print_law_report(laws: dict):
    """Pretty print the discovered laws."""
    print("\n" + "="*70)
    print("DISCOVERED PHYSICAL LAWS")
    print("="*70)
    
    # Thermodynamic Laws
    if 'second_law' in laws['discovered_laws']:
        print("\n1. THERMODYNAMIC LAWS")
        print("-" * 70)
        
        sl = laws['discovered_laws']['second_law']
        status = "[+] COMPLIANT" if sl['compliant'] else "[!] VIOLATED"
        print(f"   Second Law of Thermodynamics: {status}")
        print(f"     • Entropy rate: {sl['entropy_rate']:.6f}")
        print(f"     • Violations: {sl['violations']} ({sl['violation_rate']*100:.1f}%)")
        print(f"     • Interpretation: Universe is {sl['interpretation']}")
        
        if 'landauer_principle' in laws['discovered_laws']:
            lp = laws['discovered_laws']['landauer_principle']
            status = "[+] VERIFIED" if lp['verified'] else "[~] PARTIAL"
            print(f"\n   Landauer Principle: {status}")
            print(f"     • Heat per collapse: {lp['heat_per_collapse']:.6f}")
            print(f"     • Variation: {lp['variation']:.6f}")
            print(f"     • {lp['interpretation']}")
    
    # Conservation Laws
    if 'energy_conservation' in laws['discovered_laws']:
        print("\n2. CONSERVATION LAWS")
        print("-" * 70)
        
        ec = laws['discovered_laws']['energy_conservation']
        status = "[+] CONSERVED" if ec['conserved'] else "[!] NOT CONSERVED"
        print(f"   Energy Conservation: {status}")
        print(f"     • Mean energy: {ec['mean_energy']:.6f}")
        print(f"     • Variation: {ec['variation']*100:.2f}%")
        print(f"     • Standard deviation: {ec['std_dev']:.6f}")
        
        if 'matter_conservation' in laws['discovered_laws']:
            mc = laws['discovered_laws']['matter_conservation']
            status = "[+] CONSERVED" if mc['conserved'] else "[!] VIOLATED"
            print(f"\n   Matter Conservation: {status}")
            print(f"     • Violations: {mc['violations']}")
            print(f"     • Total created: {mc['total_created']:.6f}")
            print(f"     • {mc['interpretation']}")
    
    # Emergent Constants
    if laws['emergent_constants']:
        print("\n3. EMERGENT PHYSICAL CONSTANTS")
        print("-" * 70)
        
        if 'c_effective' in laws['emergent_constants']:
            c = laws['emergent_constants']['c_effective']
            print(f"   Speed of Light (effective):")
            print(f"     • c = {c['value']:.6f} ± {c['std']:.6f}")
            print(f"     • Stability: {(1-c['stability'])*100:.2f}%")
        
        if 'temperature_memory_coupling' in laws['emergent_constants']:
            tmc = laws['emergent_constants']['temperature_memory_coupling']
            print(f"\n   Temperature-Memory Coupling Constant:")
            print(f"     • α_TM = {tmc['alpha_TM']:.6f} ± {tmc['std']:.6f}")
            print(f"     • {tmc['interpretation']}")
        
        if 'cooling_rate' in laws['emergent_constants']:
            cr = laws['emergent_constants']['cooling_rate']
            print(f"\n   Cooling Rate Constant:")
            print(f"     • γ = {cr['gamma']:.6f}")
            print(f"     • Half-life: {cr['half_life']:.2f} time units")
            print(f"     • {cr['interpretation']}")
    
    # Correlations
    if laws['correlations']:
        print("\n4. INFORMATION-THERMODYNAMIC CORRELATIONS")
        print("-" * 70)
        
        for name, corr in laws['correlations'].items():
            sig = "[+]" if corr['significant'] else "[~]"
            print(f"   {name.replace('_', ' ').title()}:")
            print(f"     {sig} r = {corr['correlation']:.3f} (p = {corr['p_value']:.4f})")
            print(f"     • {corr['interpretation']}")
    
    # Phase Transitions
    if laws['phase_transitions']:
        print("\n5. PHASE TRANSITIONS DETECTED")
        print("-" * 70)
        for trans in laws['phase_transitions']:
            print(f"   {trans['type'].title()} Transition:")
            print(f"     • At steps: {trans['steps'][:3]}...")
            print(f"     • {trans['interpretation']}")
    
    # Summary
    if 'summary' in laws:
        print("\n6. SIMULATION SUMMARY")
        print("-" * 70)
        s = laws['summary']
        print(f"   Total steps: {s['total_steps']}")
        print(f"   Time span: {s['time_span']:.2f}")
        print(f"   Temperature range: {s['temperature_range'][0]:.3f} → {s['temperature_range'][1]:.3f}")
        print(f"   Memory growth: {s['memory_growth']:.1f}×")
        print(f"   Total collapses: {s['total_collapses']}")
        print(f"   Entropy change: {s['entropy_change']:.6f}")
    
    print("\n" + "="*70)

def run_law_discovery(steps=300, size=(64, 16)):
    """
    Run simulation and discover emergent laws.
    """
    print("="*70)
    print("AUTOMATED LAW DISCOVERY")
    print("="*70)
    print(f"\nRunning {steps} step simulation to build history...")
    print(f"Universe size: {size[0]} × {size[1]} cells")
    
    # Create and run engine
    engine = RealityEngine(size=size, dt=0.1, device='cpu')
    engine.initialize(mode='big_bang')
    
    # Evolve and suppress output
    print("\n[*] Evolving universe...")
    for i, state in enumerate(engine.evolve(steps=steps)):
        if i % (steps // 10) == 0:
            print(f"  Progress: {i}/{steps} steps ({i*100//steps}%)")
    
    print(f"\n[+] Evolution complete: {len(engine.history)} states recorded")
    
    # Discover laws
    print("\n[*] Analyzing history to discover physical laws...")
    laws = engine.discover_laws(min_history=50)
    
    if laws['status'] == 'insufficient_data':
        print(f"[!] Insufficient data: {laws['history_length']} < {laws['required']}")
        return None
    
    # Print report
    print_law_report(laws)
    
    # Save to file
    save_path = repo_root / 'examples' / 'discovered_laws.json'
    with open(save_path, 'w') as f:
        json.dump(laws, f, indent=2)
    print(f"\n[*] Full report saved to: {save_path}")
    
    return laws

if __name__ == '__main__':
    print("\n" + "="*70)
    print("DISCOVERY: Fundamental physics emerges from pure dynamics!")
    print("No preset laws - only SEC, Confluence, and thermodynamics.")
    print("="*70 + "\n")
    
    laws = run_law_discovery(steps=300, size=(64, 16))
    
    if laws:
        print("\n" + "="*70)
        print("[SUCCESS] LAW DISCOVERY COMPLETE")
        print("="*70)
        print("\nKey Findings:")
        
        # Count verified laws
        verified_count = 0
        if laws['discovered_laws'].get('second_law', {}).get('compliant'):
            verified_count += 1
            print("  [+] Second Law of Thermodynamics")
        if laws['discovered_laws'].get('landauer_principle', {}).get('verified'):
            verified_count += 1
            print("  [+] Landauer Principle (Information -> Heat)")
        if laws['discovered_laws'].get('energy_conservation', {}).get('conserved'):
            verified_count += 1
            print("  [+] Energy Conservation")
        if laws['discovered_laws'].get('matter_conservation', {}).get('conserved'):
            verified_count += 1
            print("  [+] Matter Conservation")
        
        print(f"\nTotal: {verified_count} fundamental laws verified!")
        print(f"Plus {len(laws['emergent_constants'])} emergent constants discovered.")
        print("="*70 + "\n")
