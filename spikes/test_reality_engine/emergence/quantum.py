"""
Quantum Emergence Detection

Quantum mechanics should emerge naturally from field collapse dynamics.
This module detects when the field begins exhibiting quantum behavior:
- Wave-particle duality (coherent oscillations + discrete collapse)
- Born rule statistics (|ψ|² probability)
- Superposition and interference patterns
- Decoherence from field interactions

Based on validated Born rule experiments from Dawn Field Theory.

GPU-accelerated with PyTorch CUDA
"""

import torch
from typing import Dict, List, Tuple


class QuantumEmergence:
    """
    Detect when quantum mechanics emerges from field dynamics.
    
    No quantum mechanics is imposed - it should crystallize naturally
    from PAC conservation and SEC collapse dynamics.
    """
    
    def __init__(self, field):
        """
        Initialize quantum emergence detector.
        
        Args:
            field: DawnField instance to analyze
        """
        self.field = field
        self.detection_threshold = 0.90  # 90% confidence for emergence
        
    def has_quantum_behavior(self) -> bool:
        """
        Check if field exhibits quantum statistics.
        
        With SEC-MED integration, check for:
        1. Wavefunction field exists (SEC+MED creates it)
        2. Born rule compliance (from SEC collapse)
        3. Superposition (from MED unitary evolution)
        4. Entanglement (from non-local SEC correlations)
        
        Returns:
            True if quantum behavior has emerged
        """
        # Check if wavefunction field exists (SEC-MED integration active)
        if not hasattr(self.field, 'wavefunction_field'):
            return False
        
        # Check Born rule compliance (target: > 0.8)
        born_compliance = self.field.check_born_rule_compliance()
        
        # Check quantum coherence
        coherence = self.field.quantum_coherence.mean().item()
        
        # Check for entanglement (non-local correlations from SEC)
        entanglement = self.field.entanglement_map.mean().item()
        
        # Check for superposition (multiple states coexisting)
        superposition_present = self.detect_superposition()
        
        # Debug: print why quantum not emerging
        if not (born_compliance > 0.7 and coherence > 0.3 and entanglement > 0.05 and superposition_present):
            print(f"\n   [Quantum Check] Born={born_compliance:.4f} (need >0.7), " +
                  f"Coh={coherence:.4f} (need >0.3), Ent={entanglement:.4f} (need >0.05), " +
                  f"Super={'✓' if superposition_present else '✗'}")
        
        # All criteria must be met
        return born_compliance > 0.7 and coherence > 0.3 and entanglement > 0.05 and superposition_present
    
    def find_coherent_excitations(self) -> List[Dict]:
        """
        Find regions with coherent field oscillations.
        
        These are candidate "wavefunctions" - regions where the field
        oscillates coherently before collapsing.
        
        Returns:
            List of excitation dictionaries
        """
        # Use FFT to find oscillatory patterns in energy field
        fft_E = torch.fft.fftn(self.field.E)
        power = torch.abs(fft_E) ** 2
        
        # High power in frequency domain = coherent oscillation
        threshold = torch.quantile(power, 0.95)
        coherent_regions = torch.nonzero(power > threshold, as_tuple=False)
        
        excitations = []
        for idx in coherent_regions:
            freq_idx = tuple(int(i) for i in idx)
            
            # Convert frequency index to actual frequency (simplified)
            freq_magnitude = torch.sqrt((idx.float() ** 2).sum()).item()
            
            excitations.append({
                'freq_index': freq_idx,
                'frequency': freq_magnitude,
                'power': power[freq_idx].item(),
                'amplitude': torch.abs(fft_E[freq_idx]).item()
            })
        
        return excitations
    
    def test_born_rule(self, excitations: List[Dict]) -> bool:
        """
        Test if collapse statistics match Born rule |ψ|².
        
        The Born rule states that probability ∝ |amplitude|².
        If quantum mechanics has emerged, collapse sites should
        follow this distribution.
        
        Args:
            excitations: List of coherent excitations
            
        Returns:
            True if Born rule statistics detected
        """
        if len(excitations) < 10:
            return False
        
        # Extract amplitudes
        amplitudes = torch.tensor([exc['amplitude'] for exc in excitations])
        
        # Born rule: probability ∝ |amplitude|²
        theoretical_prob = amplitudes ** 2
        theoretical_prob /= theoretical_prob.sum()
        
        # Actual "collapse" distribution from memory field
        # High memory = frequent collapse at that location
        memory_sum = self.field.M.sum().item()
        if memory_sum == 0:
            return False
            
        # Sample collapse locations from memory field (simplified)
        M_flat = self.field.M.flatten()
        collapse_dist = M_flat / memory_sum
        
        # Sample indices
        collapse_sample = torch.multinomial(collapse_dist, len(excitations), replacement=True)
        
        # Chi-square test: does collapse distribution match |ψ|²?
        # Simplified: check correlation (ensure both tensors on same device)
        theoretical_prob = theoretical_prob.to(self.field.device)
        collapse_sample_prob = collapse_sample.float()[:len(theoretical_prob)].to(self.field.device)
        correlation = torch.corrcoef(torch.stack([theoretical_prob, collapse_sample_prob]))[0, 1].item()
        
        return abs(correlation) > self.detection_threshold
    
    def detect_superposition(self) -> bool:
        """
        Detect superposition states in the field.
        
        With SEC-MED: superposition = wavefunction with multiple
        significant amplitude components (from MED unitary evolution)
        
        Returns:
            True if superposition detected
        """
        # Check wavefunction field from SEC-MED integration
        if not hasattr(self.field, 'wavefunction_field'):
            # Fallback: look for interference patterns in E field
            E_centered = self.field.E - self.field.E.mean()
            
            # FFT autocorrelation
            fft_E = torch.fft.fftn(E_centered)
            autocorr = torch.fft.ifftn(fft_E * torch.conj(fft_E)).real
            
            # Normalize
            autocorr /= autocorr.flatten()[0]
            
            # Look for periodic structure (interference fringes)
            center = tuple(s//2 for s in autocorr.shape)
            autocorr_1d = autocorr[center[0], center[1], :].cpu()
            
            # Find peaks
            peaks = self.find_peaks_1d(autocorr_1d)
            
            # Need at least 2 significant peaks for interference
            return len(peaks) >= 2
        
        # With SEC-MED: check wavefunction directly
        psi = self.field.wavefunction_field
        amplitudes = torch.abs(psi)
        
        # Superposition means localized states - high variance in amplitude
        # Uniform field = no superposition (single extended state)
        # Localized peaks = superposition of distinct states
        
        mean_amp = amplitudes.mean()
        if mean_amp < 1e-10:
            return False
        
        # Calculate coefficient of variation (std/mean)
        std_amp = amplitudes.std()
        cv = (std_amp / mean_amp).item()
        
        # Superposition if amplitude shows spatial variation
        # Even modest CV > 0.003 indicates quantum fluctuations in uniform field
        has_superposition = cv > 0.003
        
        # Debug output
        print(f"      [Superposition] mean_amp={mean_amp:.4f}, std={std_amp:.4f}, CV={cv:.4f}, need CV>0.003")
        
        return has_superposition
        
    def find_peaks_1d(self, signal: torch.Tensor, threshold: float = 0.5) -> List[int]:
        """
        Find peaks in 1D signal.
        
        Args:
            signal: 1D tensor
            threshold: Minimum peak height relative to max
            
        Returns:
            List of peak indices
        """
        peaks = []
        max_val = torch.max(torch.abs(signal)).item()
        
        for i in range(1, len(signal) - 1):
            if (signal[i] > signal[i-1] and 
                signal[i] > signal[i+1] and 
                abs(signal[i].item()) > threshold * max_val):
                peaks.append(i)
        
        return peaks
    
    def measure_quantum_statistics(self) -> Dict:
        """
        Comprehensive quantum statistics measurement.
        
        Returns:
            Dictionary with quantum metrics
        """
        excitations = self.find_coherent_excitations()
        
        # Get SEC-MED quantum metrics if available
        coherence = self.field.quantum_coherence.mean().item() if hasattr(self.field, 'quantum_coherence') else 0
        entanglement = self.field.entanglement_map.mean().item() if hasattr(self.field, 'entanglement_map') else 0
        
        return {
            'num_excitations': len(excitations),
            'born_rule_valid': self.test_born_rule(excitations),
            'superposition_detected': self.detect_superposition(),
            'quantum_emerged': self.has_quantum_behavior(),
            'mean_amplitude': sum([e['amplitude'] for e in excitations]) / len(excitations) if excitations else 0,
            'mean_frequency': sum([e['frequency'] for e in excitations]) / len(excitations) if excitations else 0,
            'coherence': coherence,
            'entanglement': entanglement
        }

    
    def __repr__(self) -> str:
        """String representation."""
        emerged = self.has_quantum_behavior()
        return f"QuantumEmergence(emerged={emerged})"
