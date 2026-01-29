"""
Experiment 07: Complete State Trace

Records EVERYTHING at every step to JSON for post-analysis.
No filtering, no aggregation - raw data.

Output: Full trace file with every node, value, connection at every timestep.
"""

import json
import numpy as np
from datetime import datetime
from reality_seed.genesis import GenesisSeed


def run_trace(n_steps=500, output_file=None):
    """
    Record complete state at every step.
    """
    print("=" * 70)
    print("COMPLETE STATE TRACE")
    print("=" * 70)
    print()
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/full_trace_{timestamp}.json"
    
    genesis = GenesisSeed(initial_value=1.0)
    genesis.ratio_memory_weight = 0.5
    
    # Complete trace
    trace = {
        'metadata': {
            'n_steps': n_steps,
            'initial_value': 1.0,
            'ratio_memory_weight': 0.5,
            'timestamp': datetime.now().isoformat(),
        },
        'steps': []
    }
    
    print("Recording %d steps..." % n_steps)
    
    for step in range(n_steps):
        # Capture state BEFORE step
        nodes_before = {
            nid: {
                'id': nid,
                'value': float(n.value),
                'neighbors': [nb.id for nb in n.neighbors]
            }
            for nid, n in genesis.substrate.nodes.items()
        }
        
        # Take step
        event = genesis.step()
        
        # Capture state AFTER step
        nodes_after = {
            nid: {
                'id': nid,
                'value': float(n.value),
                'neighbors': [nb.id for nb in n.neighbors]
            }
            for nid, n in genesis.substrate.nodes.items()
        }
        
        # Detect changes
        ids_before = set(nodes_before.keys())
        ids_after = set(nodes_after.keys())
        
        created = list(ids_after - ids_before)
        destroyed = list(ids_before - ids_after)
        
        # Record split ratios if applicable
        split_info = None
        if len(created) == 2 and len(destroyed) == 0:
            # Find parent by value conservation
            new_values = [nodes_after[nid]['value'] for nid in created]
            for pid, pnode in nodes_before.items():
                if abs(sum(new_values) - pnode['value']) < 0.0001:
                    split_info = {
                        'parent_id': pid,
                        'parent_value': pnode['value'],
                        'child_ids': created,
                        'child_values': new_values,
                        'ratio': new_values[0] / pnode['value'] if pnode['value'] > 0 else 0
                    }
                    break
        
        # Full step record
        step_record = {
            'step': step,
            'nodes': list(nodes_after.values()),
            'node_count': len(nodes_after),
            'total_value': sum(n['value'] for n in nodes_after.values()),
            'created': created,
            'destroyed': destroyed,
            'split_info': split_info,
            'event_type': event.get('type', 'unknown') if isinstance(event, dict) else 'step'
        }
        
        trace['steps'].append(step_record)
        
        if step % 100 == 0:
            print("  Step %d: %d nodes" % (step, len(nodes_after)))
    
    # Summary stats
    trace['summary'] = {
        'final_node_count': len(genesis.substrate.nodes),
        'total_splits': sum(1 for s in trace['steps'] if s['split_info'] is not None),
        'total_created': sum(len(s['created']) for s in trace['steps']),
        'total_destroyed': sum(len(s['destroyed']) for s in trace['steps']),
    }
    
    # Write to file
    print()
    print("Writing trace to: %s" % output_file)
    
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(trace, f, indent=2)
    
    # File size
    import os
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print("File size: %.2f MB" % size_mb)
    
    print()
    print("Summary:")
    print("  Steps: %d" % n_steps)
    print("  Final nodes: %d" % trace['summary']['final_node_count'])
    print("  Total splits: %d" % trace['summary']['total_splits'])
    print("  Total created: %d" % trace['summary']['total_created'])
    print("  Total destroyed: %d" % trace['summary']['total_destroyed'])
    
    return trace, output_file


if __name__ == "__main__":
    trace, filepath = run_trace(n_steps=500)
    print()
    print("Trace saved. Ready for post-analysis.")
