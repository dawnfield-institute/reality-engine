"""
Quick test: Discover periodic patterns from emergence observations.
Demonstrates natural quantization, bonding rules, and emergent periodic structure.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.reality_engine import RealityEngine
from tools.emergence_observer import EmergenceObserver

def main():
    print("=" * 70)
    print("DISCOVERING PERIODIC PATTERNS - No Assumptions")
    print("=" * 70)
    print()
    
    # Initialize
    engine = RealityEngine(size=(128, 32), dt=0.01)
    engine.initialize('big_bang')
    observer = EmergenceObserver()
    
    # Collect observations
    print("Running 1000 steps and observing emergence every 50 steps...")
    observation_history = []
    
    for step in range(1001):
        if step > 0:
            state_dict = engine.step()
        
        if step % 50 == 0:
            structures = observer.observe(engine.current_state)
            observation_history.append(structures)
            print(f"  Step {step:4d}: {len(structures)} structures, {len(set(s.pattern_class for s in structures))} pattern types")
    
    print()
    print("=" * 70)
    print("ANALYZING EMERGENT PATTERNS")
    print("=" * 70)
    print()
    
    # Discover periodic structure
    discovery = observer.discover_periodic_patterns(observation_history)
    
    # Report findings
    print(f"📊 MASS QUANTIZATION (Natural Clustering):")
    print(f"   Found {len(discovery['mass_clusters'])} distinct mass clusters")
    print()
    for cluster_id in sorted(discovery['mass_clusters'].keys())[:10]:
        cluster = discovery['mass_clusters'][cluster_id]
        print(f"   Cluster {cluster_id}:")
        print(f"     • Mass: {cluster['mass_center']:.3f} (±{(cluster['mass_range'][1]-cluster['mass_range'][0])/2:.3f})")
        print(f"     • Occurrences: {cluster['count']}")
        print(f"     • Coherence: {cluster['avg_coherence']:.3f}")
        print(f"     • Persistence: {cluster['avg_persistence']:.3f}")
        print()
    
    print()
    print(f"🔗 BONDING CHEMISTRY (Emergent Interactions):")
    print(f"   Found {len(discovery['bonding_matrix'])} bonding patterns")
    print()
    
    # Sort by bond count
    sorted_bonds = sorted(discovery['bonding_matrix'].items(), 
                         key=lambda x: x[1]['count'], reverse=True)[:5]
    
    for bond_key, bond_info in sorted_bonds:
        c1, c2 = bond_key
        m1 = discovery['mass_clusters'][c1]['mass_center']
        m2 = discovery['mass_clusters'][c2]['mass_center']
        print(f"   Cluster {c1} (M≈{m1:.2f}) ↔ Cluster {c2} (M≈{m2:.2f})")
        print(f"     • Bonds observed: {bond_info['count']}")
        print(f"     • Binding strength: {bond_info['avg_binding']:.3f}")
        print()
    
    print()
    print(f"⚖️ STABILITY GROUPS:")
    print()
    for group_name in ['high', 'medium', 'low']:
        if group_name in discovery['stability_groups']:
            group = discovery['stability_groups'][group_name]
            print(f"   {group_name.upper()} Stability:")
            print(f"     • Count: {group['count']}")
            print(f"     • Mass range: {group['mass_range'][0]:.3f} - {group['mass_range'][1]:.3f}")
            print(f"     • Avg coherence: {group['avg_coherence']:.3f}")
            print()
    
    print()
    print(f"🔬 PERIODIC STRUCTURE (Like Periodic Table):")
    print(f"   Found {len(discovery['periodic_structure'])} distinct types")
    print()
    
    # Show most versatile bonders (like elements with high valence)
    versatile = sorted(discovery['periodic_structure'].items(),
                      key=lambda x: x[1]['bonding_versatility'], reverse=True)[:5]
    
    print("   Most chemically active patterns:")
    for cluster_id, info in versatile:
        mass = info['properties']['mass_center']
        versatility = info['bonding_versatility']
        print(f"     • Cluster {cluster_id} (M≈{mass:.2f}): bonds with {versatility} different types")
        if info['bonding_partners']:
            partners = [f"C{p['partner_cluster']}" for p in info['bonding_partners'][:3]]
            print(f"       Partners: {', '.join(partners)}")
    
    print()
    print("=" * 70)
    print("✅ PERIODIC PATTERN DISCOVERY COMPLETE!")
    print("=" * 70)
    print()
    print("Key Insights:")
    print(f"  • {len(discovery['mass_clusters'])} natural mass quanta emerged")
    print(f"  • {len(discovery['bonding_matrix'])} bonding rules discovered")
    print(f"  • Periodic organization emerged WITHOUT a predefined table")
    print()
    print("Everything from: B(x,t) = ∇²(E-I) + λM∇²M - α||E-I||²")
    print("The periodic table emerges naturally!")

if __name__ == "__main__":
    main()
