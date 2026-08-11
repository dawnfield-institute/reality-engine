"""
Herniation Detection and Application for Reality Engine

Herniations are quantum collapse events where information-energy field clashes
create classical reality from potential fields. Each collapse is a "tick" of
emergent time, actualizing quantum potential into observable phenomena.

The herniation mechanism is THE key to reality emergence in the Reality Engine.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional


class HerniationDetector:
    """
    Detects and applies herniation events in field dynamics.
    
    A herniation occurs when:
    - Information (I) and energy (E) fields clash with sufficient pressure
    - Field gradients are strong enough to break field coherence
    - Memory hasn't already collapsed the region (M < threshold)
    
    The collapse transfers quantum potential → classical actuality,
    creating observable reality and ticking emergent time forward.
    """
    
    def __init__(
        self,
        device: str = 'cpu',
        threshold_sigma: float = 2.0,
        collapse_strength: float = 0.01,
        memory_rate: float = 0.01,
        turbulence_scale: float = 0.005,
        collapse_radius: float = 3.0,
        max_collapses_per_step: int = 50
    ):
        """
        Initialize herniation detector.
        
        Args:
            device: Compute device ('cpu' or 'cuda')
            threshold_sigma: Stddev multiplier for herniation threshold
            collapse_strength: Strength of quantum→classical transfer (0.01 = gentle)
            memory_rate: Rate of memory accumulation at collapse sites
            turbulence_scale: Scale of random perturbations to maintain dynamics
            collapse_radius: Spatial extent of collapse kernel (in grid units)
            max_collapses_per_step: Maximum herniation sites to process per step
        """
        self.device = device
        self.threshold_sigma = threshold_sigma
        self.collapse_strength = collapse_strength
        self.memory_rate = memory_rate
        self.turbulence_scale = turbulence_scale
        self.collapse_radius = collapse_radius
        self.max_collapses_per_step = max_collapses_per_step
        
        # Statistics tracking
        self.total_herniations = 0
        self.herniation_history = []
        
    def detect(
        self,
        energy: torch.Tensor,
        information: torch.Tensor,
        memory: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Detect herniation sites where quantum collapse should occur.
        
        Herniation potential = pressure × gradient_magnitude × (1 - memory)
        
        Where:
        - pressure = information × energy (field clash intensity)
        - gradient = spatial rate of change (field instability)
        - (1-M) = uncollapsed field (new regions more likely to herniate)
        
        Args:
            energy: Actual energy field (E) [H, W]
            information: Potential information field (I) [H, W]
            memory: Memory field (M) [H, W]
            
        Returns:
            Dictionary containing:
            - sites: Tensor of (y, x) coordinates where herniations occur
            - rate: Herniations per unit area
            - intensity: Average herniation potential at sites
            - collapse_probability: Quantum coherence-based collapse probability
            - count: Number of herniation sites
        """
        with torch.no_grad():
            # Calculate field pressure (information-energy clash)
            pressure = information * energy
            
            # Calculate spatial gradients (field instability)
            grad_y = torch.diff(pressure, dim=0, prepend=pressure[-1:, :])
            grad_x = torch.diff(pressure, dim=1, prepend=pressure[:, -1:])
            gradient_magnitude = torch.sqrt(grad_y**2 + grad_x**2)
            
            # Herniation potential: where pressure + gradients exceed boundary strength
            # (1-M) factor: already-collapsed regions less likely to herniate again
            # Normalize components to prevent explosion
            pressure_normalized = torch.tanh(pressure / 10.0)  # Bound to [-1, 1]
            gradient_normalized = torch.tanh(gradient_magnitude / 10.0)  # Bound to [0, 1]
            herniation_potential = pressure_normalized * gradient_normalized * (1.0 - memory / 100.0)
            
            # Dynamic threshold based on field statistics
            mean_potential = herniation_potential.mean()
            std_potential = herniation_potential.std()
            threshold = mean_potential + self.threshold_sigma * std_potential
            
            # Find herniation sites
            herniation_mask = herniation_potential > threshold
            herniation_sites = torch.nonzero(herniation_mask)
            
            if len(herniation_sites) > 0:
                # Calculate statistics
                herniation_rate = len(herniation_sites) / (energy.shape[0] * energy.shape[1])
                avg_intensity = herniation_potential[herniation_mask].mean().item()
                
                # Quantum collapse probability based on field coherence
                # High coherence (ordered fields) → higher collapse probability
                coherence = torch.abs(torch.fft.fft2(pressure)).mean()
                collapse_probability = 1.0 / (1.0 + torch.exp(-coherence / 1000.0))
                
                result = {
                    'sites': herniation_sites,
                    'rate': herniation_rate,
                    'intensity': avg_intensity,
                    'collapse_probability': collapse_probability.item(),
                    'count': len(herniation_sites)
                }
                
                # Track in history
                self.herniation_history.append({
                    'iteration': self.total_herniations,
                    'count': len(herniation_sites),
                    'rate': herniation_rate,
                    'intensity': avg_intensity
                })
                if len(self.herniation_history) > 1000:
                    self.herniation_history.pop(0)
                
                self.total_herniations += len(herniation_sites)
                
                return result
            
            # No herniations detected
            return {
                'sites': torch.tensor([], device=self.device),
                'rate': 0.0,
                'intensity': 0.0,
                'collapse_probability': 0.0,
                'count': 0
            }
    
    def apply(
        self,
        herniations: Dict[str, Any],
        state
    ):
        """
        Apply herniation collapses to actualize reality.
        
        Each collapse:
        1. Transfers potential (I) → actual (E): quantum → classical
        2. Increases memory (M): marks region as collapsed
        3. Raises temperature (T): adds energy to system
        4. Adds turbulence: maintains field gradients for future herniations
        
        The turbulence is critical - without it, collapses would smooth out
        all gradients and the system would freeze. The turbulence "stirs the pot"
        to keep reality dynamics active.
        
        Args:
            herniations: Dictionary from detect() containing sites and metadata
            state: RealityEngine state object with .actual, .potential, .memory, .temperature
        """
        if herniations['count'] == 0:
            return
        
        sites = herniations['sites']
        
        # Limit collapses per step for performance
        max_collapses = min(self.max_collapses_per_step, len(sites))
        sites = sites[:max_collapses]
        
        # Get device from state fields
        field_device = state.actual.device
        collapse_prob = float(herniations['collapse_probability'])
        
        with torch.no_grad():
            # Process each herniation site
            for site in sites:
                y, x = site[0].item(), site[1].item()
                
                # Create spatial collapse kernel (Gaussian)
                Y, X = torch.meshgrid(
                    torch.arange(state.actual.shape[0], device=field_device),
                    torch.arange(state.actual.shape[1], device=field_device),
                    indexing='ij'
                )
                
                # Distance with Möbius topology wrapping
                dy = torch.minimum(
                    torch.abs(Y - y),
                    state.actual.shape[0] - torch.abs(Y - y)
                )
                dx = torch.minimum(
                    torch.abs(X - x),
                    state.actual.shape[1] - torch.abs(X - x)
                )
                dist = torch.sqrt(dy.float()**2 + dx.float()**2)
                
                # Gaussian collapse kernel
                collapse_kernel = torch.exp(-dist**2 / (2 * self.collapse_radius**2))
                
                # Collapse amount scaled by probability and strength
                collapse_amount = collapse_kernel * collapse_prob * self.collapse_strength
                
                # 1. QUANTUM → CLASSICAL TRANSFER
                # Transfer information to energy (actualization of potential)
                transfer = collapse_amount * state.potential
                state.potential.sub_(transfer)  # Reduce potential (in-place)
                state.actual.add_(transfer)     # Increase actual (in-place)
                
                # 2. MEMORY ACCUMULATION
                # Mark region as collapsed to reduce future herniation probability
                state.memory.add_(collapse_amount * self.memory_rate)
                
                # 3. TEMPERATURE INCREASE
                # Collapses add energy, raising local temperature
                # Use small increment to prevent thermal runaway
                state.temperature.add_(collapse_kernel * 0.1)
                
                # 4. TURBULENCE INJECTION
                # Add random perturbations to maintain field gradients
                # This is CRITICAL - without turbulence, system freezes
                noise = torch.randn_like(collapse_kernel) * self.turbulence_scale
                state.potential.add_(collapse_kernel * noise)
                state.actual.add_(collapse_kernel * noise * 0.5)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current herniation statistics"""
        if not self.herniation_history:
            return {
                'total_herniations': 0,
                'avg_rate': 0.0,
                'avg_intensity': 0.0,
                'recent_count': 0
            }
        
        recent = self.herniation_history[-100:]  # Last 100 events
        
        return {
            'total_herniations': self.total_herniations,
            'avg_rate': np.mean([h['rate'] for h in recent]),
            'avg_intensity': np.mean([h['intensity'] for h in recent]),
            'recent_count': sum(h['count'] for h in recent)
        }
    
    def reset_statistics(self):
        """Reset herniation tracking statistics"""
        self.total_herniations = 0
        self.herniation_history = []
