"""SimulationConfig — all tuneable parameters in one place."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class SimulationConfig:
    """Central configuration for a simulation run.

    Grid geometry, physical constants, operator toggles, and dashboard settings.
    """

    # --- grid geometry ---------------------------------------------------
    nu: int = 128
    nv: int = 32

    # --- physical constants (derived from DFT, not tuneable) -------------
    xi: float = 1.0571072          # Ξ — universal balance constant (1 + π/55)
    phi: float = 1.618033988749895  # φ — golden ratio
    alpha_pac: float = 0.964       # PAC memory coefficient
    lambda_freq: float = 0.020     # λ — universal frequency (Hz)

    # --- evolution parameters (adaptive can override) --------------------
    dt: float = 0.001              # Base time step
    field_scale: float = 50.0      # Soft clamp boundary
    noise_scale: float = 0.01      # Thermal noise amplitude fraction
    t_min: float = 0.1             # Temperature floor
    t_max: float = 10.0            # Temperature ceiling

    # --- memory dynamics -------------------------------------------------
    mass_gen_coeff: float = 0.63   # Mass generation coefficient
    quantum_pressure_coeff: float = 0.015
    mass_diffusion_coeff: float = 0.002

    # --- confluence ------------------------------------------------------
    confluence_weight: float = 0.3  # Blend factor for confluence step
    confluence_every: int = 1       # Apply confluence every N ticks

    # --- operator toggles ------------------------------------------------
    enable_thermal_noise: bool = True
    enable_confluence: bool = True
    enable_adaptive: bool = True
    enable_normalization: bool = True

    # --- dashboard -------------------------------------------------------
    dashboard_port: int = 8050
    dashboard_update_every: int = 5  # Send state to dashboard every N ticks

    # --- device ----------------------------------------------------------
    device: Optional[torch.device] = None

    def get_device(self) -> torch.device:
        if self.device is not None:
            return self.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
