"""
Adaptive Parameter Controller for Reality Engine

Uses QBE balance and PAC residuals to automatically tune all simulation parameters.
NO manual tuning - parameters self-organize based on field dynamics.

Inspired by CIMM's quantum potential layer and PACEngine's conservation feedback.
Key insight: QBE feedback DIRECTLY modulates parameters for conservation.
"""

import torch
import numpy as np
from typing import Dict, Optional
from collections import deque


class AdaptiveParameters:
    """
    Auto-balances simulation parameters using QBE-driven feedback.
    
    Following CIMM's quantum potential layer approach:
    - QBE residual directly modulates gamma (like QPL modulates learning rate)
    - Uses momentum and memory smoothing for stability
    - Applies damping to prevent oscillations
    - PAC is secondary check, QBE drives primary adaptation
    """
    
    def __init__(
        self,
        initial_gamma: float = 0.01,
        initial_dt: float = 0.01,
        min_gamma: float = 0.001,
        max_gamma: float = 0.5,
        min_dt: float = 0.0001,
        max_dt: float = 0.01,  # Capped for stability
        history_length: int = 100
    ):
        """
        Initialize adaptive parameter controller with QBE feedback.
        
        Following CIMM's quantum potential layer approach where QBE
        directly modulates parameters for conservation.
        """
        # Current parameter values
        self.gamma = initial_gamma
        self.dt = initial_dt
        
        # Bounds (safety only)
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.min_dt = min_dt
        self.max_dt = max_dt
        
        # History for feedback
        self.pac_residuals = deque(maxlen=history_length)
        self.qbe_residuals = deque(maxlen=history_length)
        self.field_energies = deque(maxlen=history_length)
        self.nan_events = deque(maxlen=history_length)
        
        # QBE momentum tracking (like CIMM's qpl_memory)
        self.qbe_momentum = 0.0
        self.gamma_memory = initial_gamma
        
        # Adaptation rates (following CIMM's gentle 0.03-0.06 adjustments)
        self.qbe_adapt_rate = 0.05   # 5% QBE-driven adjustment
        self.dt_adapt_rate = 0.05    # 5% dt adjustment
        self.damping_factor = 0.98   # From CIMM's QPL damping
        
        # Targets
        self.qbe_target = 0.01  # Target QBE residual (~1%)
        self.target_energy_stability = 0.2  # <20% energy change per step
        
        # NaN recovery mode
        self.nan_recovery_steps = 0
        
    def update(
        self,
        pac_residual: float,
        qbe_residual: float,
        field_energy: float,
        had_nan: bool
    ) -> Dict[str, float]:
        """
        Update parameters using QBE feedback like CIMM's quantum potential layer.
        
        Key insight from CIMM: QBE feedback directly modulates learning rates
        and parameters to maintain conservation.
        
        Args:
            pac_residual: PAC conservation error (secondary check)
            qbe_residual: QBE constraint residual (primary driver)
            field_energy: Total field energy for dt adaptation
            had_nan: Whether NaN occurred (emergency signal)
        
        Returns:
            Dictionary with current parameter values and diagnostics
        """
        # Record history
        self.pac_residuals.append(pac_residual)
        self.qbe_residuals.append(qbe_residual)
        self.field_energies.append(field_energy)
        self.nan_events.append(1.0 if had_nan else 0.0)
        
        # Need some history before adapting
        if len(self.qbe_residuals) < 10:
            return self._current_params()
        
        # QBE-driven gamma adaptation (like CIMM's QPL tuning)
        self._adapt_gamma_qbe(qbe_residual, pac_residual, had_nan)
        
        # Adapt dt based on energy growth rate
        self._adapt_dt(field_energy, had_nan)
        
        return self._current_params()
    
    def _adapt_gamma_qbe(self, qbe_residual: float, pac_residual: float, had_nan: bool):
        """
        Adapt gamma using QBE feedback, following CIMM's quantum potential layer.
        
        From CIMM quantum_potential_layer.py:
        - QBE feedback directly modulates parameters
        - Uses momentum and memory for smooth changes
        - Applies damping to prevent oscillations
        
        This is the KEY difference: we use QBE (not PAC) as primary driver!
        """
        if had_nan:
            # Emergency increase like CIMM's sharp deviation handling
            self.gamma = min(self.gamma * 1.5, self.max_gamma)
            self.gamma_memory = self.gamma
            self.nan_recovery_steps = 50
            return
        
        # Recovery mode: gradually relax back to normal
        if self.nan_recovery_steps > 0:
            self.nan_recovery_steps -= 1
            return
        
        # Compute QBE feedback (normalized residual from target)
        qbe_feedback = (qbe_residual - self.qbe_target) / (self.qbe_target + 1e-10)
        
        # Clip to prevent extreme adjustments (CIMM uses similar bounds)
        qbe_feedback = np.clip(qbe_feedback, -1.0, 1.0)
        
        # Update gamma with QBE modulation (like CIMM's learning rate adjustment)
        # From CIMM: adaptive_controller.learning_rate *= (1 + 0.06 * qbe_feedback)
        gamma_adjustment = 1.0 + self.qbe_adapt_rate * qbe_feedback
        
        # Apply adjustment with momentum smoothing
        new_gamma = self.gamma * gamma_adjustment
        
        # Blend with memory for stability (CIMM uses 0.85 memory, 0.15 new)
        self.gamma_memory = 0.85 * self.gamma_memory + 0.15 * new_gamma
        self.gamma = 0.87 * self.gamma + 0.13 * self.gamma_memory
        
        # Apply damping to prevent oscillations
        self.gamma *= self.damping_factor
        
        # Secondary check: if PAC is drifting badly, override with gentle increase
        if pac_residual > 0.1:  # 10% PAC drift - emergency!
            self.gamma *= 1.02  # Gentle 2% increase
        
        # Apply bounds
        self.gamma = np.clip(self.gamma, self.min_gamma, self.max_gamma)
        
        # Update QBE momentum for next step
        self.qbe_momentum = 0.9 * self.qbe_momentum + 0.1 * qbe_feedback
    
    def _adapt_dt(self, field_energy: float, had_nan: bool):
            # Don't print every time
        
        # Apply bounds
        self.gamma = np.clip(self.gamma, self.min_gamma, self.max_gamma)
    
    def _adapt_dt(self, field_energy: float, had_nan: bool):
        """
        Adapt time step dt based on energy growth rate.
        
        Strategy:
        - If energy growing too fast: DECREASE dt
        - If energy stable: INCREASE dt (faster simulation)
        - If NaN: DECREASE dt immediately
        """
        if had_nan:
            # Immediate decrease if unstable
            self.dt *= (1.0 - 2.0 * self.dt_adapt_rate)
            print(f"  [ADAPTIVE] NaN detected - decreasing dt to {self.dt:.5f}")
            return
        
        if len(self.field_energies) < 20:
            return
        
        # Compute energy growth rate
        recent_energies = list(self.field_energies)[-20:]
        energy_growth = (recent_energies[-1] - recent_energies[0]) / recent_energies[0]
        
        if abs(energy_growth) > self.target_energy_stability:
            # Energy changing too fast - reduce dt
            self.dt *= (1.0 - self.dt_adapt_rate)
            # Don't print every time, too noisy
        else:
            # Energy stable - can safely increase dt for speed
            self.dt *= (1.0 + 0.5 * self.dt_adapt_rate)
        
        # Apply bounds
        self.dt = np.clip(self.dt, self.min_dt, self.max_dt)
    
    def _current_params(self) -> Dict[str, float]:
        """Return current parameter values"""
        return {
            'gamma': self.gamma,
            'dt': self.dt,
            'pac_quality': self._compute_pac_quality(),
            'stability_score': self._compute_stability_score()
        }
    
    def _compute_pac_quality(self) -> float:
        """Compute PAC conservation quality (0=bad, 1=perfect)"""
        if len(self.pac_residuals) < 10:
            return 0.5
        
        recent_residuals = list(self.pac_residuals)[-10:]
        mean_residual = np.mean(recent_residuals)
        
        # Exponential decay: 1.0 for zero residual, approaches 0 for large
        return np.exp(-mean_residual * 10)
    
    def _compute_stability_score(self) -> float:
        """Compute numerical stability score (0=unstable, 1=stable)"""
        if len(self.nan_events) < 10:
            return 0.5
        
        recent_nans = list(self.nan_events)[-10:]
        nan_rate = np.mean(recent_nans)
        
        # 1.0 if no NaNs, 0.0 if all NaNs
        return 1.0 - nan_rate
    
    def get_diagnostics(self) -> Dict[str, any]:
        """Return diagnostic information for monitoring"""
        return {
            'gamma': self.gamma,
            'dt': self.dt,
            'pac_quality': self._compute_pac_quality(),
            'stability_score': self._compute_stability_score(),
            'pac_residual_mean': np.mean(list(self.pac_residuals)) if self.pac_residuals else 0.0,
            'pac_residual_std': np.std(list(self.pac_residuals)) if self.pac_residuals else 0.0,
            'energy_mean': np.mean(list(self.field_energies)) if self.field_energies else 0.0,
            'nan_rate': np.mean(list(self.nan_events)) if self.nan_events else 0.0,
        }
