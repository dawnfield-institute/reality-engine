"""FusionOperator — stellar nucleosynthesis.

When mass and temperature both exceed thresholds at the same point,
fusion occurs: mass converts to energy and produces metallicity (Z).

    fusion_rate = η · M · T · σ(M - M_ign) · σ(T - T_ign) · 1/(1 + Z/Z_p)
    dM = -fusion_rate · dt              (mass consumed)
    dE = dI = α·fusion_rate·dt / 2      (PAC-conserving energy release)
    dZ = +fusion_rate · dt · ζ          (metals produced, ζ = yield fraction)

The sigmoid σ enforces ignition thresholds. The Z poisoning gate creates
natural star death: as Z accumulates, fusion slows → star cools → dies.
No star → no fusion → no heavy elements → no atoms heavier than hydrogen.
This is THE causal gate in the emergence chain.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus


class FusionOperator:
    """Stellar fusion: mass + heat → energy + metals."""

    def __init__(
        self,
        eta: float = 0.1,           # fusion rate coefficient
        mass_ignition: float = 3.0,  # mass threshold for ignition
        temp_ignition: float = 2.0,  # temperature threshold for ignition
        efficiency: float = 0.5,     # mass → energy conversion efficiency
        metal_yield: float = 0.1,    # fraction of consumed mass → metals
        sharpness: float = 5.0,      # sigmoid sharpness at thresholds
        z_poisoning: float = 5.0,    # Z level at which fusion halves (iron ceiling)
    ) -> None:
        self.eta = eta
        self.mass_ignition = mass_ignition
        self.temp_ignition = temp_ignition
        self.efficiency = efficiency
        self.metal_yield = metal_yield
        self.sharpness = sharpness
        self.z_poisoning = z_poisoning

    @property
    def name(self) -> str:
        return "fusion"

    @torch.no_grad()
    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        M, T, E, Z = state.M, state.T, state.E, state.Z
        dt = config.dt

        # Soft ignition gates (sigmoid → 0 below threshold, → 1 above)
        mass_gate = torch.sigmoid(self.sharpness * (M - self.mass_ignition))
        temp_gate = torch.sigmoid(self.sharpness * (T - self.temp_ignition))

        # Metallicity poisoning: fusion efficiency drops as Z accumulates.
        # At Z = z_poisoning, rate halves. At Z >> z_poisoning, rate → 0.
        # This is the iron ceiling — you can't fuse past iron.
        z_gate = 1.0 / (1.0 + Z / self.z_poisoning)

        # Fusion rate: only nonzero where mass, temp, and fuel are sufficient
        fusion_rate = self.eta * M * T * mass_gate * temp_gate * z_gate

        # Mass consumed, energy released (PAC-conserving), metals produced
        # PAC = E + I + α·M.  If dM = -rate·dt, then for conservation:
        # dE + dI = α·rate·dt (split equally between E and I)
        dM = -fusion_rate * dt
        energy_release = fusion_rate * dt
        dE = energy_release * 0.5
        dI = energy_release * 0.5
        dZ = fusion_rate * dt * self.metal_yield

        M_new = torch.clamp(M + dM, min=0.0)
        E_new = E + dE
        I_new = state.I + dI
        Z_new = Z + dZ

        total_fused = fusion_rate.sum().item() * dt

        metrics = dict(state.metrics)
        metrics["fusion_rate_total"] = total_fused
        metrics["metallicity_mean"] = Z_new.mean().item()

        if bus is not None and total_fused > 1e-6:
            bus.emit("fusion_occurred", {
                "total_fused": total_fused,
                "metallicity_mean": Z_new.mean().item(),
            })

        return state.replace(M=M_new, E=E_new, I=I_new, Z=Z_new, metrics=metrics)
