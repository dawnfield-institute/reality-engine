"""
Conservation Analyzer - Detect and measure conservation laws.

Tracks:
- Energy conservation (E+I)
- Momentum conservation
- Angular momentum conservation
- Information conservation
- PAC functional conservation
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from base_analyzer import BaseAnalyzer, Detection


class ConservationAnalyzer(BaseAnalyzer):
    """
    Analyzes conservation laws in the emergent physics.
    
    Measures how well various quantities are conserved over time
    and detects which conservation laws emerge naturally.
    """
    
    def __init__(self, min_confidence: float = 0.9):
        """
        Initialize conservation analyzer.
        
        Args:
            min_confidence: Minimum confidence for reporting conservation
                          (conservation laws should be very confident!)
        """
        super().__init__("conservation_analyzer", min_confidence)
        
        # Track quantities over time
        self.total_energy = []
        self.total_momentum = []
        self.total_angular_momentum = []
        self.pac_functional = []
        
    def analyze(self, state: Dict) -> List[Detection]:
        """Analyze conservation laws in current state"""
        detections = []
        current_time = state.get('time', 0.0)
        
        # Extract fields
        A = self._to_numpy(state.get('actual'))
        P = self._to_numpy(state.get('potential'))
        M = self._to_numpy(state.get('memory'))
        
        if A is None or P is None or M is None:
            return detections
        
        # Measure total quantities
        total_E_plus_I = float(np.sum(A + P))
        pac_functional_value = float(np.sum(P + A + 0.964 * M))
        
        self.total_energy.append({
            'time': current_time,
            'value': total_E_plus_I
        })
        self.pac_functional.append({
            'time': current_time,
            'value': pac_functional_value
        })
        
        # Need history to detect conservation
        if len(self.total_energy) < 50:
            return detections
        
        # Analyze E+I conservation
        energy_conservation = self._check_conservation(
            self.total_energy[-50:],
            "energy_plus_info",
            "E + I = constant"
        )
        if energy_conservation:
            detections.append(energy_conservation)
        
        # Analyze PAC conservation
        pac_conservation = self._check_conservation(
            self.pac_functional[-50:],
            "PAC_functional",
            "P + A + α·M = constant"
        )
        if pac_conservation:
            detections.append(pac_conservation)
        
        # Momentum conservation (if structures have velocities)
        if 'structures' in state:
            momentum_cons = self._check_momentum_conservation(
                state['structures'], current_time
            )
            if momentum_cons:
                detections.append(momentum_cons)
        
        return detections
    
    def _to_numpy(self, field) -> Optional[np.ndarray]:
        """Convert field to numpy array"""
        if field is None:
            return None
        if torch.is_tensor(field):
            return field.cpu().numpy() if field.is_cuda else field.numpy()
        return np.array(field)
    
    def _check_conservation(self, history: List[Dict], 
                           quantity_name: str, 
                           equation: str) -> Optional[Detection]:
        """
        Check if a quantity is conserved over time.
        
        Returns Detection if conserved with high confidence.
        """
        values = np.array([h['value'] for h in history])
        times = np.array([h['time'] for h in history])
        
        # Measure variation relative to mean
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if abs(mean_val) < 1e-10:
            return None
        
        # Relative variation
        relative_variation = std_val / abs(mean_val)
        
        # Conservation confidence: 1.0 for no variation, decreases with variation
        # Allow up to 5% variation for "conserved"
        confidence = max(0.0, 1.0 - relative_variation / 0.05)
        
        if confidence > self.min_confidence:
            # Check for systematic drift (trend over time)
            if len(times) > 10:
                slope, _, r_value, _, _ = stats.linregress(times, values)
                drift_rate = abs(slope) / (abs(mean_val) + 1e-10)
                
                # Penalize for systematic drift
                confidence *= np.exp(-drift_rate * 100)
            
            return Detection(
                type=f"{quantity_name}_conservation",
                confidence=confidence,
                time=times[-1],
                equation=equation,
                parameters={
                    'mean_value': float(mean_val),
                    'std_deviation': float(std_val),
                    'relative_variation': float(relative_variation),
                    'drift_rate': float(drift_rate) if len(times) > 10 else 0.0
                },
                properties={
                    'conserved': True,
                    'variation_percent': float(relative_variation * 100),
                    'measurements': len(values)
                }
            )
        
        return None
    
    def _check_momentum_conservation(self, structures: List, 
                                    current_time: float) -> Optional[Detection]:
        """Check if total momentum is conserved"""
        # Need structures with velocity
        structures_with_v = [s for s in structures 
                           if hasattr(s, 'velocity') and s.velocity is not None]
        
        if len(structures_with_v) < 2:
            return None
        
        # Calculate total momentum
        total_p = np.zeros(2)
        for s in structures_with_v:
            v = np.array(s.velocity)
            total_p += s.mass * v
        
        momentum_mag = np.linalg.norm(total_p)
        
        self.total_momentum.append({
            'time': current_time,
            'value': momentum_mag,
            'components': total_p
        })
        
        # Need history
        if len(self.total_momentum) < 30:
            return None
        
        # Check conservation
        recent = self.total_momentum[-30:]
        values = np.array([h['value'] for h in recent])
        
        mean_p = np.mean(values)
        std_p = np.std(values)
        
        if mean_p < 1e-10:
            # Zero momentum conserved (system at rest in COM frame)
            confidence = 1.0 - std_p
        else:
            relative_var = std_p / mean_p
            confidence = max(0.0, 1.0 - relative_var / 0.05)
        
        if confidence > self.min_confidence:
            return Detection(
                type="momentum_conservation",
                confidence=confidence,
                time=current_time,
                equation="Σ(m·v) = constant",
                parameters={
                    'total_momentum': float(momentum_mag),
                    'std_deviation': float(std_p)
                },
                properties={
                    'conserved': True,
                    'num_structures': len(structures_with_v)
                }
            )
        
        return None
    
    def get_conservation_summary(self) -> Dict:
        """Get summary of all conservation laws"""
        summary = {}
        
        # E+I conservation
        if self.total_energy:
            values = [h['value'] for h in self.total_energy]
            summary['energy_plus_info'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'relative_variation': float(np.std(values) / (abs(np.mean(values)) + 1e-10)),
                'conserved': np.std(values) / (abs(np.mean(values)) + 1e-10) < 0.05
            }
        
        # PAC conservation
        if self.pac_functional:
            values = [h['value'] for h in self.pac_functional]
            summary['PAC_functional'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'relative_variation': float(np.std(values) / (abs(np.mean(values)) + 1e-10)),
                'conserved': np.std(values) / (abs(np.mean(values)) + 1e-10) < 0.05
            }
        
        # Momentum conservation
        if self.total_momentum:
            values = [h['value'] for h in self.total_momentum]
            summary['momentum'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'relative_variation': float(np.std(values) / (abs(np.mean(values)) + 1e-10)) if np.mean(values) > 1e-10 else 0.0
            }
        
        return summary
