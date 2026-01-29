"""
Reality Seed Visualization

Generic visualization that doesn't assume what patterns mean.
Shows the graph, shows the numbers, lets human interpret.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Dict
import networkx as nx

from .genesis import GenesisSeed, GenesisObserver


class GenesisVisualizer:
    """
    Visualizes genesis without interpreting it.
    
    Shows:
    - The graph (nodes, edges)
    - Value distribution
    - Component structure
    - Raw statistics
    
    Does NOT label things as "matter" or "gravity" etc.
    Human watches and interprets.
    """
    
    def __init__(self, genesis: GenesisSeed):
        self.genesis = genesis
        self.observer = GenesisObserver(genesis)
        self.fig = None
        self.axes = None
        
    def setup_dashboard(self):
        """Create the observation dashboard."""
        self.fig = plt.figure(figsize=(16, 10))
        
        # Grid: 2 rows, 3 columns
        # [0,0] Main graph  | [0,1] Value histogram | [0,2] Component sizes
        # [1,0] Split ratios | [1,1] Trends         | [1,2] Raw stats
        
        self.axes = {
            'graph': self.fig.add_subplot(2, 3, 1),
            'values': self.fig.add_subplot(2, 3, 2),
            'components': self.fig.add_subplot(2, 3, 3),
            'splits': self.fig.add_subplot(2, 3, 4),
            'trends': self.fig.add_subplot(2, 3, 5),
            'stats': self.fig.add_subplot(2, 3, 6),
        }
        
        self.fig.suptitle('Reality Seed - Genesis Observation', fontsize=14)
        plt.tight_layout()
        
    def update(self, frame: int = None) -> List:
        """Update visualization with current state."""
        # Run some steps
        self.genesis.run(10)
        
        # Observe
        obs = self.observer.observe()
        
        # Clear all axes
        for ax in self.axes.values():
            ax.clear()
        
        # === Graph visualization ===
        self._draw_graph(self.axes['graph'])
        
        # === Value distribution ===
        values = [n.value for n in self.genesis.substrate.nodes.values()]
        if values:
            self.axes['values'].hist(values, bins=30, alpha=0.7, color='blue')
            self.axes['values'].axvline(np.mean(values), color='red', 
                                        linestyle='--', label=f'mean={np.mean(values):.4f}')
        self.axes['values'].set_title('Value Distribution')
        self.axes['values'].set_xlabel('Value')
        self.axes['values'].set_ylabel('Count')
        self.axes['values'].legend()
        
        # === Component sizes ===
        sizes = obs.get('component_sizes', [])
        if sizes:
            self.axes['components'].bar(range(len(sizes)), sizes, alpha=0.7)
        self.axes['components'].set_title(f'Components (n={obs.get("n_components", 0)})')
        self.axes['components'].set_xlabel('Component')
        self.axes['components'].set_ylabel('Size')
        
        # === Split ratios ===
        ratios = self.genesis.split_ratios
        if ratios:
            self.axes['splits'].plot(ratios, 'b.', alpha=0.3, markersize=2)
            # Rolling mean
            if len(ratios) > 20:
                window = min(50, len(ratios) // 2)
                rolling = np.convolve(ratios, np.ones(window)/window, mode='valid')
                self.axes['splits'].plot(range(window-1, len(ratios)), 
                                        rolling, 'b-', linewidth=2)
            self.axes['splits'].axhline(0.618, color='gold', linestyle='--', 
                                        label='1/phi = 0.618')
            self.axes['splits'].axhline(0.5, color='gray', linestyle=':', 
                                        label='0.5')
            self.axes['splits'].set_ylim(0, 1)
            self.axes['splits'].legend(loc='upper right')
        self.axes['splits'].set_title('Split Ratios')
        self.axes['splits'].set_xlabel('Event')
        self.axes['splits'].set_ylabel('Ratio')
        
        # === Trends ===
        if len(self.observer.observations) > 2:
            events = [o['event'] for o in self.observer.observations]
            n_nodes = [o['n_nodes'] for o in self.observer.observations]
            n_comps = [o['n_components'] for o in self.observer.observations]
            
            ax2 = self.axes['trends'].twinx()
            self.axes['trends'].plot(events, n_nodes, 'b-', label='Nodes')
            ax2.plot(events, n_comps, 'r-', label='Components')
            self.axes['trends'].set_xlabel('Event')
            self.axes['trends'].set_ylabel('Nodes', color='blue')
            ax2.set_ylabel('Components', color='red')
            self.axes['trends'].legend(loc='upper left')
            ax2.legend(loc='upper right')
        self.axes['trends'].set_title('Growth')
        
        # === Raw stats ===
        conserved_str = '[OK]' if obs['conservation']['conserved'] else '[!!]'
        stats_text = f"""
Event: {obs['event']}
Nodes: {obs['n_nodes']}
Components: {obs['n_components']}
Conservation: {conserved_str}

Value Stats:
  mean: {obs.get('value_stats', {}).get('mean', 0):.4f}
  max: {obs.get('value_stats', {}).get('max', 0):.4f}
  gini: {obs.get('value_stats', {}).get('gini', 0):.4f}

Depth:
  mean: {obs.get('depth', {}).get('mean', 0):.2f}
  max: {obs.get('depth', {}).get('max', 0)}

Split Convergence:
  mean ratio: {obs.get('split_convergence', {}).get('mean_ratio', 0):.4f}
  phi_inv dist: {obs.get('split_convergence', {}).get('distance_from_phi_inv', 0):.4f}
"""
        self.axes['stats'].text(0.05, 0.95, stats_text, 
                               transform=self.axes['stats'].transAxes,
                               fontsize=9, verticalalignment='top',
                               fontfamily='monospace')
        self.axes['stats'].axis('off')
        self.axes['stats'].set_title('Raw Observations')
        
        plt.tight_layout()
        return list(self.axes.values())
    
    def _draw_graph(self, ax):
        """Draw the node graph using networkx."""
        substrate = self.genesis.substrate
        
        # Build networkx graph
        G = nx.DiGraph()
        
        # Sample if too large
        max_nodes = 200
        nodes_to_show = list(substrate.nodes.keys())
        if len(nodes_to_show) > max_nodes:
            # Show nodes with highest value
            nodes_by_value = sorted(substrate.nodes.values(), 
                                   key=lambda n: n.value, reverse=True)
            nodes_to_show = [n.id for n in nodes_by_value[:max_nodes]]
        
        nodes_set = set(nodes_to_show)
        
        for nid in nodes_to_show:
            node = substrate.nodes[nid]
            G.add_node(nid, value=node.value)
            
            # Add edges
            for cid in node.children:
                if cid in nodes_set:
                    G.add_edge(nid, cid, type='child')
            for nbid in node.neighbors:
                if nbid in nodes_set and nid < nbid:
                    G.add_edge(nid, nbid, type='neighbor')
        
        if len(G.nodes) == 0:
            ax.text(0.5, 0.5, 'No nodes yet', ha='center', va='center')
            ax.set_title('Graph')
            return
        
        # Layout
        try:
            pos = nx.spring_layout(G, k=2/np.sqrt(len(G.nodes)), iterations=50)
        except:
            pos = nx.random_layout(G)
        
        # Node sizes by value
        values = [substrate.nodes[nid].value * 100 + 10 for nid in G.nodes]
        
        # Draw
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=values, 
                              node_color=values, cmap='viridis', alpha=0.7)
        
        # Edges by type
        child_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'child']
        neighbor_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'neighbor']
        
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=child_edges, 
                              edge_color='gray', alpha=0.3, arrows=True)
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=neighbor_edges,
                              edge_color='red', alpha=0.3, style='dashed')
        
        ax.set_title(f'Graph ({len(G.nodes)} nodes shown)')
        ax.axis('off')
    
    def animate(self, n_frames: int = 100, interval: int = 200):
        """Run animated visualization."""
        self.setup_dashboard()
        anim = FuncAnimation(self.fig, self.update, frames=n_frames,
                            interval=interval, blit=False)
        plt.show()
        return anim
    
    def snapshot(self, n_steps: int = 1000):
        """Run n_steps and show final state."""
        self.genesis.run(n_steps)
        self.setup_dashboard()
        self.update()
        plt.show()
        
        return self.observer.observe()


def run_genesis(n_steps: int = 1000, animate: bool = False):
    """
    Main entry point.
    
    Just runs genesis and observes. No interpretation.
    """
    print("="*60)
    print("REALITY SEED - GENESIS")
    print("="*60)
    print()
    print("Running PAC dynamics. Watching what happens.")
    print("No pre-defined physics. No expected patterns.")
    print()
    
    genesis = GenesisSeed(initial_value=1.0)
    viz = GenesisVisualizer(genesis)
    
    if animate:
        viz.animate(n_frames=n_steps // 10)
    else:
        obs = viz.snapshot(n_steps=n_steps)
        
        print("\n" + "="*60)
        print("FINAL OBSERVATION")
        print("="*60)
        
        print(f"\nNodes: {obs['n_nodes']}")
        print(f"Components: {obs['n_components']}")
        conserved = obs['conservation']['conserved']
        print(f"Conservation: {'[OK] CONSERVED' if conserved else '[!!] VIOLATED'}")
        
        if 'value_stats' in obs:
            print(f"\nValue distribution:")
            print(f"  mean: {obs['value_stats']['mean']:.6f}")
            print(f"  gini: {obs['value_stats']['gini']:.4f}")
        
        if 'split_convergence' in obs:
            print(f"\nSplit convergence:")
            print(f"  mean ratio: {obs['split_convergence']['mean_ratio']:.4f}")
            print(f"  distance from phi_inv: {obs['split_convergence']['distance_from_phi_inv']:.4f}")
        
        trends = viz.observer.get_trends()
        if 'error' not in trends:
            print(f"\nTrends:")
            for k, v in trends.items():
                print(f"  {k}: {v:.4f}")
    
    return genesis, viz


if __name__ == "__main__":
    run_genesis(n_steps=2000, animate=False)
