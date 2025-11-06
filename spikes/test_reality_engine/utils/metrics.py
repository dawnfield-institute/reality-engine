"""
Emergence Metrics and Validation

Track what's emerging and validate against known physics.
Key validations from Dawn Field Theory experiments:
- 96.4% PAC conservation correlation
- 0.020 Hz universal resonance frequency  
- Born rule quantum statistics
- 1/r gravity emergence

GPU-accelerated with PyTorch CUDA
"""

import torch
from typing import Dict, List


class EmergenceMetrics:
    """
    Track emergent phenomena and validate against known physics.
    
    This class monitors:
    - PAC conservation (target: 96.4% correlation)
    - Universal resonance (target: 0.020 Hz)
    - Quantum statistics emergence
    - Particle formation
    - Force law emergence
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.conservation_history = []
        self.energy_history = []
        self.info_history = []
        self.memory_history = []
        self.herniation_history = []
        
        # Validation targets from experiments
        self.target_conservation_corr = 0.964  # 96.4%
        self.target_resonance_freq = 0.020     # 0.020 Hz
        
    def check_conservation(self, field) -> Dict:
        """
        Check PAC conservation law.
        
        Total E + I + M should be conserved.
        Target: 96.4% correlation from experiments.
        
        Args:
            field: DawnField instance
            
        Returns:
            Conservation metrics
        """
        total = (field.E.sum() + field.I.sum() + field.M.sum()).item()
        self.conservation_history.append(total)
        
        self.energy_history.append(field.E.sum().item())
        self.info_history.append(field.I.sum().item())
        self.memory_history.append(field.M.sum().item())
        
        if len(self.conservation_history) > 1:
            # Check stability
            recent = self.conservation_history[-100:] if len(self.conservation_history) > 100 else self.conservation_history
            recent_tensor = torch.tensor(recent)
            std = recent_tensor.std().item()
            mean = recent_tensor.mean().item()
            
            # Correlation with initial state
            if len(self.conservation_history) > 10:
                initial = self.conservation_history[0]
                history_tensor = torch.tensor(self.conservation_history)
                initial_tensor = torch.full_like(history_tensor, initial)
                correlation = torch.corrcoef(torch.stack([history_tensor, initial_tensor]))[0, 1].item()
            else:
                correlation = 1.0
            
            return {
                'total': total,
                'stability': std / mean if mean != 0 else float('inf'),
                'conserved': std / mean < 0.05,  # 5% threshold
                'correlation': correlation,
                'target_met': abs(correlation) >= self.target_conservation_corr,
                'energy_fraction': field.E.sum().item() / total if total != 0 else 0,
                'info_fraction': field.I.sum().item() / total if total != 0 else 0,
                'memory_fraction': field.M.sum().item() / total if total != 0 else 0
            }
        
        return {
            'total': total,
            'stability': 0,
            'conserved': True,
            'correlation': 1.0,
            'target_met': True,
            'energy_fraction': field.E.sum().item() / total if total != 0 else 0,
            'info_fraction': field.I.sum().item() / total if total != 0 else 0,
            'memory_fraction': field.M.sum().item() / total if total != 0 else 0
        }
    
    def detect_resonance(self, history: List[float] = None, 
                        target_freq: float = None) -> Dict:
        """
        Check for universal resonance frequency in evolution.
        
        Target: 0.020 Hz from experiments
        
        Args:
            history: Time series to analyze (defaults to conservation history)
            target_freq: Target frequency (defaults to 0.020 Hz)
            
        Returns:
            Resonance detection results
        """
        if history is None:
            history = self.conservation_history
            
        if target_freq is None:
            target_freq = self.target_resonance_freq
        
        if len(history) < 100:
            return {
                'detected': False,
                'reason': 'insufficient_data',
                'length': len(history)
            }
        
        # FFT to find dominant frequency
        history_tensor = torch.tensor(history, dtype=torch.float32)
        fft = torch.fft.fft(history_tensor)
        freqs = torch.fft.fftfreq(len(history))
        power = torch.abs(fft) ** 2
        
        # Find peak (skip DC component)
        half_len = len(power) // 2
        peak_idx = torch.argmax(power[1:half_len]).item() + 1
        peak_freq = abs(freqs[peak_idx].item())
        peak_power = power[peak_idx].item()
        
        # Check if matches target
        freq_match = abs(peak_freq - target_freq) < 0.005
        
        return {
            'detected': freq_match,
            'detected_freq': peak_freq,
            'target_freq': target_freq,
            'peak_power': peak_power,
            'match': freq_match,
            'error': abs(peak_freq - target_freq),
            'confidence': peak_power / power.mean().item()
        }
    
    def analyze_emergence_timeline(self, field, quantum_emerged: bool, 
                                  particles_emerged: bool) -> Dict:
        """
        Analyze when different phenomena emerged.
        
        Args:
            field: DawnField instance
            quantum_emerged: Has quantum mechanics emerged?
            particles_emerged: Have particles emerged?
            
        Returns:
            Timeline analysis
        """
        # Calculate herniation rate
        if len(field.herniation_history) > 100:
            recent_herniations = torch.tensor(field.herniation_history[-100:], dtype=torch.float32)
            herniation_rate = recent_herniations.mean().item()
        else:
            herniation_rate = 0
            
        return {
            'current_time': field.time,
            'current_step': field.step_count,
            'quantum_emerged': quantum_emerged,
            'particles_emerged': particles_emerged,
            'total_herniations': sum(field.herniation_history),
            'herniation_rate': herniation_rate
        }
    
    def validate_gravity_emergence(self, field) -> Dict:
        """
        Check if 1/r gravity potential has emerged.
        
        Should appear in fractal dispersion patterns from collapses.
        
        Args:
            field: DawnField instance
            
        Returns:
            Gravity validation results
        """
        # Find position with maximum energy
        max_energy_idx = torch.argmax(field.E.flatten()).item()
        
        # Manual unravel for 3D
        shape = field.E.shape
        z_idx = max_energy_idx // (shape[0] * shape[1])
        remainder = max_energy_idx % (shape[0] * shape[1])
        y_idx = remainder // shape[0]
        x_idx = remainder % shape[0]
        
        center = torch.tensor([x_idx, y_idx, z_idx], dtype=torch.float32)
        
        # Sample radial profile
        distances = []
        potentials = []
        
        for r in range(1, 20):
            # Sample points at distance r
            samples = self.sample_sphere(field.E, center, r)
            
            if len(samples) > 0:
                distances.append(r)
                samples_tensor = torch.stack(samples)
                potentials.append(samples_tensor.mean().item())
        
        if len(distances) < 5:
            return {'emerged': False, 'reason': 'insufficient_samples'}
        
        # Fit to 1/r law
        distances = torch.tensor(distances, dtype=torch.float32)
        potentials = torch.tensor(potentials, dtype=torch.float32)
        
        # Log-log fit: log(V) = log(A) - log(r)
        # Slope should be -1 for 1/r
        try:
            log_dist = torch.log(distances)
            log_pot = torch.log(potentials + 1e-10)
            
            # Manual linear regression: slope = cov(x,y) / var(x)
            x_mean = log_dist.mean()
            y_mean = log_pot.mean()
            cov = ((log_dist - x_mean) * (log_pot - y_mean)).sum()
            var = ((log_dist - x_mean) ** 2).sum()
            slope = (cov / var).item()
            
            # Check if close to -1
            is_inverse = abs(slope + 1.0) < 0.3  # 30% tolerance
            
            return {
                'emerged': is_inverse,
                'slope': slope,
                'expected_slope': -1.0,
                'error': abs(slope + 1.0),
                'distances': distances.tolist(),
                'potentials': potentials.tolist()
            }
        except:
            return {'emerged': False, 'reason': 'fit_failed'}
    
    def sample_sphere(self, field: torch.Tensor, center: torch.Tensor, 
                     radius: int) -> List[torch.Tensor]:
        """
        Sample field values on a sphere.
        
        Args:
            field: 3D field tensor
            center: Center point tensor
            radius: Sphere radius
            
        Returns:
            List of sampled tensors
        """
        samples = []
        
        # Sample points approximately at radius
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    dist = torch.sqrt(torch.tensor(dx**2 + dy**2 + dz**2, dtype=torch.float32))
                    
                    if abs(dist.item() - radius) < 1.0:  # Within 1 unit of target radius
                        x = (int(center[0].item()) + dx) % field.shape[0]
                        y = (int(center[1].item()) + dy) % field.shape[1]
                        z = (int(center[2].item()) + dz) % field.shape[2]
                        
                        samples.append(field[x, y, z])
        
        return samples
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive emergence report.
        
        Returns:
            Full metrics report
        """
        # Calculate stability if enough data
        if len(self.conservation_history) > 100:
            recent_values = torch.tensor(self.conservation_history[-100:], dtype=torch.float32)
            stability = (recent_values.std() / recent_values.mean()).item()
        else:
            stability = 0
            
        report = {
            'conservation': {
                'history_length': len(self.conservation_history),
                'current_total': self.conservation_history[-1] if self.conservation_history else 0,
                'stability': stability
            },
            'resonance': self.detect_resonance() if len(self.conservation_history) >= 100 else {'detected': False},
            'timeseries': {
                'conservation': self.conservation_history.copy(),
                'energy': self.energy_history.copy(),
                'information': self.info_history.copy(),
                'memory': self.memory_history.copy()
            }
        }
        
        return report
    
    def __repr__(self) -> str:
        """String representation."""
        return f"EmergenceMetrics(samples={len(self.conservation_history)})"
