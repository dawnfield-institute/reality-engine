"""
SpectralAnalyzer — measure Ξ from the field's spectral decomposition.

On a Möbius band:
    Periodic (symmetric) modes:        eigenvalues  ~ n²
    Antiperiodic (antisymmetric) modes: eigenvalues  ~ (n + ½)²

Ξ  =  E_antisym / E_sym     (weighted energy ratio)

The confluence operator *is* the symmetry operation, so it serves
double duty: time-stepper AND measurement tool.

If the topology is doing its job, the energy ratio should converge
to ~1.0571 without injecting that number.
"""

from __future__ import annotations

import torch

from ..dynamics.confluence import ConfluenceOperator


class SpectralAnalyzer:
    """FFT-based spectral measurement using Möbius symmetry decomposition."""

    def __init__(self, confluence: ConfluenceOperator, n_modes: int = 32) -> None:
        self.confluence = confluence
        self.n_modes = n_modes

    # ------------------------------------------------------------------
    # Core Ξ measurement
    # ------------------------------------------------------------------
    def compute_xi(
        self,
        field: torch.Tensor,
        n_modes: int | None = None,
    ) -> float:
        """
        Compute Ξ from spectral energy ratio.

        1. Decompose field using confluence as the symmetry operation.
        2. FFT each component along u (angular dimension).
        3. Weighted energy ratio  =  Ξ.

        Parameters
        ----------
        field : (n_u, n_v) tensor
        n_modes : int, optional  — override default mode count

        Returns
        -------
        xi : float
            E_antisym / E_sym  (≈ 1.057 if topology works)
        """
        nm = n_modes or self.n_modes
        n_u = field.shape[0]
        max_n = min(nm, n_u // 2)

        # Decompose using confluence as the symmetry operation
        f_sym, f_antisym = self.confluence.decompose(field)

        # FFT each component along u (angular)
        F_sym = torch.fft.fft(f_sym, dim=0)
        F_anti = torch.fft.fft(f_antisym, dim=0)

        # Power spectra, averaged over v
        pow_sym = (F_sym.real ** 2 + F_sym.imag ** 2).mean(dim=1)
        pow_anti = (F_anti.real ** 2 + F_anti.imag ** 2).mean(dim=1)

        # Weighted energy sums
        # Symmetric modes: eigenvalues n²  (integer modes, skip DC)
        E_sym = 0.0
        for n in range(1, max_n + 1):
            E_sym += pow_sym[n].item() * n ** 2

        # Antisymmetric modes: eigenvalues (n + ½)²
        E_anti = 0.0
        for n in range(max_n):
            E_anti += pow_anti[n].item() * (n + 0.5) ** 2

        if E_sym < 1e-14:
            return 1.0  # degenerate — no periodic energy

        return E_anti / E_sym

    # ------------------------------------------------------------------
    # Full decomposition (for visualisation / diagnostics)
    # ------------------------------------------------------------------
    def decompose(
        self, field: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(f_sym, f_antisym)`` for diagnostic visualisation."""
        return self.confluence.decompose(field)

    def power_spectrum(
        self, field: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Return symmetric and antisymmetric 1-D power spectra
        (averaged over the v-dimension).
        """
        f_sym, f_anti = self.confluence.decompose(field)
        F_sym = torch.fft.fft(f_sym, dim=0)
        F_anti = torch.fft.fft(f_anti, dim=0)

        return {
            "sym": (F_sym.real ** 2 + F_sym.imag ** 2).mean(dim=1),
            "anti": (F_anti.real ** 2 + F_anti.imag ** 2).mean(dim=1),
        }
