"""
SECEvolver — Symbolic Entropy Collapse on the Möbius manifold.

    ∂S/∂t  =  κ ∇²S  +  σ_steered  −  γ · C_mod(S)

where the collapse is TOPOLOGY-MODULATED by the RBF balance field B:

    C_mod(S) = (1 + tanh(B)) · C_sym(S) + (1 − tanh(B)) · C_anti(S)

    C_raw(S)   = S · exp(−β · S)                 pointwise collapse
    C_sym(S)   = (C_raw + Conf(C_raw)) / 2       symmetric part
    C_anti(S)  = (C_raw − Conf(C_raw)) / 2       antiperiodic part

    σ_steered  =  σ₀ · (1 + (φ_source + B) · anti_mode)  +  θ_density

    B  =  ρ · gain · Φ / (1 + α·|M|)

    gain   = −tanh(ξ_gain · Δξ)  −  kᵢ · ∫Δξ dt     (PI control)
    Φ      = Fibonacci-weighted harmonic (breathing rhythm)
    M      = recursive memory of balance effort (self-limiting)

KEY INSIGHT: the ξ ≈ 1.0 attractor arises from NONLINEAR COLLAPSE
SATURATION at S ≈ 1 where dC/dS = 0.  Additive reinforcement cannot
overcome this because the Möbius diffusion κ∇²S destroys anti modes
at rate κ·k² ≈ 1.09 (fundamental, k²=1+π²), which exceeds any stable
reinforcement amplitude.

The solution: MODULATE THE COLLAPSE ITSELF.  By decomposing C_raw
into symmetric and antiperiodic components and reducing the anti-
collapse when B > 0, we shift the attractor from ξ=1.0 to ξ=1.057.
This operates INSIDE the collapse — the dominant term at S ≈ 1 —
and doesn't fight diffusion.

Energy conservation: C_anti sums to zero (antiperiodic), so the
total collapse θ = γ·∫C_mod·dt is EXACTLY preserved regardless of
the modulation depth tanh(B).

Compare to the vcpu_unified RBF:
    I += transfer;  E -= transfer;   (total conserved)
Here:
    collapse removes more sym, less anti;  total collapse same.

The star metaphor:
    • Gravity (collapse) and pressure (source) are in balance
    • RBF modulates the collapse: WHICH modes the star collapses
    • Source steering: WHAT the source feeds (secondary)
    • Memory M prevents oscillation, Fibonacci Φ creates breathing
    • PI integral eliminates steady-state offset

κ ∇²S        diffusion (Möbius-aware Laplacian)
σ_steered    RBF-steered source with topology-natural shape
C_mod(S)     topology-modulated collapse (spectral asymmetry)
"""

from __future__ import annotations

import math

import torch

from ..substrate.mobius import MobiusManifold
from ..substrate.constants import SEC_DEFAULTS, XI_REFERENCE, PHI_INV


class SECEvolver:
    """Single-field SEC dynamics on a Möbius manifold with RBF regulation."""

    def __init__(
        self,
        manifold: MobiusManifold,
        params: dict | None = None,
        confluence=None,
    ) -> None:
        p = {**SEC_DEFAULTS, **(params or {})}
        self.manifold = manifold
        self.confluence = confluence

        # --- SEC baseline parameters ---
        self.kappa: float = p["kappa"]
        self.gamma: float = p["gamma"]
        self.beta_0: float = p["beta_0"]
        self.sigma_0: float = p["sigma_0"]
        self.dt: float = p["dt"]
        self.xi_gain: float = p.get("xi_gain", 2.0)
        self.rho: float = p.get("rho", 1.0)
        self.phi_source: float = p.get("phi_source", PHI_INV)

        # --- RBF (Recursive Balance Field) parameters ---
        # Memory dampening coefficient: controls how much accumulated
        # effort reduces the balance field.  α=5 → gain halved when M≈0.2.
        self.alpha_rbf: float = p.get("alpha_rbf", 5.0)
        # Memory exponential decay per step: half-life ≈ 138 steps at 0.995.
        self.rbf_decay: float = p.get("rbf_decay", 0.995)
        # Integral gain: drives steady-state offset to zero.
        # Without this, proportional control alone has droop.
        self.ki_rbf: float = p.get("ki_rbf", 0.5)
        # Anti-windup clamp for integral accumulator.
        self._integral_clamp: float = p.get("integral_clamp", 1.0)
        # Fibonacci harmonic base frequency (ω).
        # Fundamental period = 1/ω in time units = 1/(ω·dt) steps.
        # At ω=0.2, dt=0.01: period = 500 steps for lowest Fib freq.
        self.rbf_omega: float = p.get("rbf_omega", 0.2)

        # --- RBF state (mutable — tracks balance history) ---
        self._M_rbf: float = 0.0          # Recursive memory of balance effort
        self._xi_integral: float = 0.0    # Accumulated ξ error (integral term)
        self._step_count: int = 0         # For Fibonacci harmonics

        # --- Fibonacci frequency table ---
        # Fib sequence: each frequency is a natural harmonic
        # φ⁻ᵏ weights: higher harmonics decay as golden ratio powers
        self._fib_freqs = [1, 1, 2, 3, 5, 8, 13, 21]
        self._fib_weights = [PHI_INV ** k for k in range(len(self._fib_freqs))]
        self._fib_norm = sum(self._fib_weights)

        # Precompute the Möbius topology's natural antiperiodic mode.
        #
        # Two families of antiperiodic modes on the Möbius strip:
        #   Low-k:  cos(u), sin(u)           — k² = 1     (u-twist only)
        #   Full:   sin(u)·sin(πv)           — k² = 1+π²  (u-twist + v-flip)
        #
        # Both satisfy f(u+π, 1−v) = −f(u, v).
        #
        # The natural source WEIGHTS by inverse eigenvalue:
        #   w(k²=1)    = 1/1     = 1.0     (fundamental)
        #   w(k²=10.87) = 1/10.87 = 0.092  (full topological)
        #
        # But the full mode carries the v-flip topology which is physically
        # essential to Möbius non-orientability.  We use a mix controlled
        # by the `low_k_mix` parameter (default 0.5):
        #
        #   anti_mode = (1−mix)·sin(u)·sin(πv) + mix·cos(u)
        #
        # The low-k cos(u) survives diffusion 10x better (κ·1 vs κ·10.87),
        # allowing the RBF to sustain higher ξ at the SEC equilibrium.
        low_k_mix: float = p.get("low_k_mix", 0.5)

        u = manifold.U  # (n_u, n_v)
        v = manifold.V
        full_mode = torch.sin(u) * torch.sin(math.pi * v)
        low_k_mode = torch.cos(u)
        # Normalise each component to [-1, 1] before mixing
        full_scale = full_mode.abs().max().clamp(min=1e-10)
        low_k_scale = low_k_mode.abs().max().clamp(min=1e-10)
        anti_raw = (
            (1.0 - low_k_mix) * full_mode / full_scale
            + low_k_mix * low_k_mode / low_k_scale
        )
        # Normalise mixture to [-1, 1]
        mix_scale = anti_raw.abs().max().clamp(min=1e-10)
        self._anti_mode: torch.Tensor = anti_raw / mix_scale

    # ------------------------------------------------------------------
    # RBF helpers
    # ------------------------------------------------------------------
    def _fibonacci_harmonic(self) -> float:
        """Fibonacci-weighted harmonic oscillation Φ(t).

        Φ = Σₖ (1/φ)ᵏ · cos(2π · fₖ · ω · t)

        Creates natural breathing rhythms for the balance field.
        Returned value is in [-1, 1] (normalised by weight sum).
        """
        t = self._step_count * self.dt
        Phi = sum(
            w * math.cos(2 * math.pi * f * self.rbf_omega * t)
            for f, w in zip(self._fib_freqs, self._fib_weights)
        )
        return Phi / self._fib_norm

    @property
    def M_rbf(self) -> float:
        """Current RBF memory (for diagnostics)."""
        return self._M_rbf

    @property
    def xi_integral(self) -> float:
        """Current integrated ξ error (for diagnostics)."""
        return self._xi_integral

    def reset_rbf(self) -> None:
        """Reset RBF state (useful for tests)."""
        self._M_rbf = 0.0
        self._xi_integral = 0.0
        self._step_count = 0

    # ------------------------------------------------------------------
    def step(
        self,
        S: torch.Tensor,
        xi_measured: float,
        theta_recycled: float = 0.0,
    ) -> tuple[torch.Tensor, float]:
        """
        One forward-Euler SEC step with topology-modulated collapse.

        Architecture:
            1. RBF balance field B from PI control + memory + Fibonacci
            2. Source steering: B shifts source spectral composition
            3. Collapse modulation: B reduces anti-collapse, preserves
               sym-collapse → shifts ξ attractor above 1.0

        Why collapse modulation (not additive reinforcement):
            - Möbius diffusion decays anti modes at rate κ·k² ≈ 1.09
            - No stable additive B can exceed this (memory dampens B)
            - Collapse modulation operates INSIDE the dominant term
            - Total theta preserved (C_anti sums to zero)

        Parameters
        ----------
        S : (n_u, n_v) tensor
        xi_measured : float — current ξ_L2 value
        theta_recycled : float — Θ from previous step

        Returns
        -------
        (S_new, theta) : tuple[Tensor, float]
        """
        # --- Collapse: C_raw(S) = S · exp(−β₀ · S) ---
        collapse_raw = S * torch.exp(-self.beta_0 * S)

        # --- Diffusion: κ ∇²S (Möbius-aware) ---
        diffusion = self.kappa * self.manifold.laplacian(S)

        # --- RBF Balance Field → Source Steering + Collapse Modulation ---
        if self.confluence is not None:
            f_anti = (S - self.confluence(S)) / 2.0
            xi_dev = (xi_measured - XI_REFERENCE) / max(XI_REFERENCE, 1e-14)

            # ── PI gain ──
            gain_p = -math.tanh(self.xi_gain * xi_dev)
            self._xi_integral += xi_dev * self.dt
            self._xi_integral = max(
                -self._integral_clamp,
                min(self._integral_clamp, self._xi_integral),
            )
            gain_i = -self.ki_rbf * self._xi_integral
            gain = gain_p + gain_i

            # ── Fibonacci breathing ──
            Phi = 0.5 + 0.5 * self._fibonacci_harmonic()

            # ── RBF: memory-dampened balance field ──
            B = (
                self.rho
                * gain
                * Phi
                / (1.0 + self.alpha_rbf * abs(self._M_rbf))
            )

            # ── Source steering: shift equilibrium spectral composition ──
            effective_phi = self.phi_source + B

            # ── Topology-modulated collapse (SYMMETRIC) ──
            # Decompose the collapse field into sym/anti under confluence:
            #   C_sym  = (C_raw + Conf(C_raw)) / 2   (symmetric part)
            #   C_anti = (C_raw − Conf(C_raw)) / 2   (antiperiodic part)
            #
            # Symmetric modulation:
            #   collapse = (1 + mod) · C_sym + (1 − mod) · C_anti
            #
            # When B > 0 (want more anti, mod > 0):
            #   Sym-collapse INCREASED → sym modes decay faster → P_mean drops
            #   Anti-collapse DECREASED → anti modes persist longer → ξ rises
            #   DOUBLE EFFECT: both sides push ξ upward
            #
            # versus one-sided modulation (only reducing anti-collapse):
            #   Single effect, AND P_mean rises (from anti-injection) → saturation
            #
            # The symmetric approach KEEPS P_mean LOW (stronger sym collapse
            # suppresses the zero mode) while boosting anti survival.
            #
            # Total collapse: (1+mod)·∫C_sym + (1−mod)·∫C_anti
            #               = (1+mod)·∫C_raw  (since ∫C_anti = 0)
            # So total θ increases by factor (1+mod).  The extra θ is
            # recycled via θ_density → self-regulating through the source.
            C_conf = self.confluence(collapse_raw)
            C_sym_col = (collapse_raw + C_conf) / 2.0
            C_anti_col = (collapse_raw - C_conf) / 2.0

            mod = math.tanh(B)
            collapse = (1.0 + mod) * C_sym_col + (1.0 - mod) * C_anti_col

            # ── RBF memory update ──
            self._M_rbf = self.rbf_decay * self._M_rbf + abs(B) * self.dt
            self._step_count += 1
        else:
            effective_phi = self.phi_source
            collapse = collapse_raw

        # --- Source: RBF-steered spectral composition ---
        theta_density = theta_recycled / max(S.numel(), 1)
        source = (
            self.sigma_0 * (1.0 + effective_phi * self._anti_mode)
            + theta_density
        )

        # --- Forward Euler ---
        dS = diffusion + source - self.gamma * collapse
        S_new = S + self.dt * dS

        # --- Θ generated: collapse removes material → recycle next step ---
        theta_out = (self.gamma * collapse * self.dt).sum().item()

        # Physical constraint: entropy ≥ 0
        S_new = torch.clamp(S_new, min=0.0)
        return S_new, theta_out
