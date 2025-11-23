"""
PAC Conservation via Global Rescaling

Simple approach: After each RBF+QBE step, rescale fields to conserve total PAC.
This is EMERGENCE, not enforcement - we're just ensuring the total doesn't drift.
"""

import torch
import numpy as np


class PACRescaler:
    """
    Ensures PAC conservation via global rescaling.
    
    After RBF+QBE dynamics update fields, this rescales them proportionally
    to maintain PAC_total = P + A + α*M = constant.
    
    This is the MINIMAL intervention - just preventing drift, not constraining structure.
    """
    
    def __init__(self, alpha_pac: float = 0.964):
        """
        Args:
            alpha_pac: PAC coefficient for memory (0.964 from experiments)
        """
        self.alpha_pac = alpha_pac
        self.initial_pac = None
        
    def initialize(self, P: torch.Tensor, A: torch.Tensor, M: torch.Tensor):
        """Record initial PAC total"""
        self.initial_pac = self._compute_pac(P, A, M)
        
    def _compute_pac(self, P: torch.Tensor, A: torch.Tensor, M: torch.Tensor) -> float:
        """Compute total PAC functional"""
        return (P.sum() + A.sum() + self.alpha_pac * M.sum()).item()
    
    def rescale(
        self, 
        P: torch.Tensor, 
        A: torch.Tensor, 
        M: torch.Tensor,
        drift_threshold: float = 0.001  # Strict: rescale if >0.1% drift
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Rescale fields to conserve PAC strictly from the beginning.
        
        PAC conservation is a FUNDAMENTAL LAW - enforce it tightly.
        
        Args:
            drift_threshold: Only rescale if PAC drift exceeds this (default 0.1%)
        
        Returns:
            (P_rescaled, A_rescaled, M_rescaled)
        """
        if self.initial_pac is None:
            self.initialize(P, A, M)
            return P, A, M
        
        current_pac = self._compute_pac(P, A, M)
        
        # Compute drift
        drift = abs(current_pac - self.initial_pac) / abs(self.initial_pac + 1e-10)
        
        # Enforce strict conservation
        if drift < drift_threshold:
            return P, A, M  # Already conserved!
        
        # Compute rescaling factor
        if abs(current_pac) < 1e-10:
            # PAC is zero or nearly zero - can't rescale
            return P, A, M
        
        scale_factor = self.initial_pac / current_pac
        
        # Rescale all fields proportionally
        P_rescaled = P * scale_factor
        A_rescaled = A * scale_factor
        M_rescaled = M * scale_factor
        
        # Verify conservation
        final_pac = self._compute_pac(P_rescaled, A_rescaled, M_rescaled)
        residual = abs(final_pac - self.initial_pac) / abs(self.initial_pac + 1e-10)
        
        print(f"  [PAC RESCALE] Drift {drift:.2%} -> {residual:.2%}, scale={scale_factor:.4f}")
        
        return P_rescaled, A_rescaled, M_rescaled
    
    def get_drift(self, P: torch.Tensor, A: torch.Tensor, M: torch.Tensor) -> float:
        """Get current PAC drift as percentage"""
        if self.initial_pac is None:
            return 0.0
        current_pac = self._compute_pac(P, A, M)
        return abs(current_pac - self.initial_pac) / abs(self.initial_pac + 1e-10)
