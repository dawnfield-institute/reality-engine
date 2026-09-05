"""A particle substrate for v3 — the engineering half of the structure problem.

The field engine cannot produce connectivity. Measured many ways over a day: the overdense
set fragments into ~1300 pieces where a web is one, and nothing — loss, clamping, quantum
pressure, gradient or curl transport, more room, more time — moves it. It has no persistent
objects, so the law detector finds no force law, because there is nothing stable enough to
have a force *between*.

`gravity_from_maxwell_pac/exp_09` already produces a cosmic web: 5000 particles, finite-range
gravity `exp(-r/r0)/r`, SEC entropy pressure. Void 50%, filament 12%, clustering 0.54,
P(k) slope -1.73 — 85% match to the observed matter spectrum, with no 1/r^2 anywhere.

This is that mechanism rebuilt against the v3 architecture: an operator protocol, an explicit
state object, a conserved ledger, and the calibrated instruments from `structure.py`. Not a
port of era-1 code into a substrate it was not written for — exp_09 is current corpus, it is
the reference implementation of local PAC gravity, and it works.

**The point of the substrate change is identity.** A cell in a density field has a value; a
particle has a trajectory. Persistence, objecthood and connectivity are properties of things
that endure, and a field of independent values has none of them. That is also what makes the
law detector usable here: particles can be tracked without inference, so a force law can be
fitted rather than hoped for.

Velocity is not "programming F = ma". A particle carries momentum the way a field carries a
value — it is what the substrate is, not a law imposed on it. What the *force* looks like is
left to the operators, and the law detector is asked afterwards what exponent actually appears.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Protocol

import torch

try:  # named constants — never bare literals (CLAUDE.md: "say which Xi you mean")
    from fracton.constants import LN2, PHI, XI_ANALYTIC
except ImportError:  # pragma: no cover — mirrors of fracton.constants.mathematical
    LN2 = math.log(2)
    PHI = (1 + math.sqrt(5)) / 2
    XI_ANALYTIC = 0.5772156649015329 + math.log((1 + math.sqrt(5)) / 2)


# ======================================================================================
# State
# ======================================================================================

@dataclass(frozen=True)
class ParticleState:
    """Immutable, mirroring FieldState. `replace()` returns a new one."""

    pos: torch.Tensor            # (N, 2) positions on a periodic box
    vel: torch.Tensor            # (N, 2)
    mass: torch.Tensor           # (N,)
    entropy: torch.Tensor        # (N,)   SEC local entropy
    box: float
    metrics: dict = field(default_factory=dict)

    # --- emergent local time ---------------------------------------------------------
    # tau is the LOCAL CLOCK RATE, a dimensionless multiplier on dt, normalised to mean 1.
    # A particle's own elapsed time is proper_time; the spread between particles is the
    # whole phenomenon. None until a LocalTime operator runs, so the substrate behaves
    # exactly as before when it does not.
    tau: Optional[torch.Tensor] = None            # (N,)
    proper_time: Optional[torch.Tensor] = None    # (N,)
    prev_delta: Optional[torch.Tensor] = None     # (N,) last tick's overdensity, for "rate"

    # --- the step ---------------------------------------------------------------------
    # The global timestep the Integrator actually took on the previous tick. Operators that
    # run BEFORE the Integrator (SECUpdate) apply their per-tick constants as rates against
    # it, so an adaptive step cannot change the physics by changing how often it fires.
    # None on the first tick and on the unrepaired path: callers fall back to config.dt.
    dt_last: Optional[float] = None
    # Acceleration accumulated by the force operators THIS tick and consumed by the
    # Integrator, which applies the kick. Force operators do not touch `vel`: the step they
    # would kick with is not known until every force has been evaluated, and choosing it
    # from this tick's forces (rather than last tick's energy) is what makes the step honest.
    acc: Optional[torch.Tensor] = None            # (N, d)
    acc_gravity: Optional[torch.Tensor] = None    # (N, d) this tick's gravity part of acc
    acc_pressure: Optional[torch.Tensor] = None   # (N, d) this tick's pressure part of acc
    # Per-particle gravitational interaction energy sum_j u_ij (each pair counted from both
    # ends), from the SAME kernel the force uses -- see gravity_potential_table. Written by
    # LocalGravity, read by PACLedger; None on a pipeline without gravity.
    potential_i: Optional[torch.Tensor] = None    # (N,)

    @property
    def n(self) -> int:
        return self.pos.shape[0]

    @property
    def device(self):
        return self.pos.device

    def replace(self, **kw) -> "ParticleState":
        return replace(self, **kw)


@dataclass
class Cosmology:
    """Scale factor evolving under a two-component Friedmann equation.

        H(a) = H0 * sqrt(Omega_m a^-3 + Omega_Lambda),   Omega_m = 1 - Omega_Lambda

    `omega_lambda` is the knob the whole experiment turns on. DFT predicts **1/phi =
    0.6180** from the PAC/SEC equilibrium (exp_25, which also puts the universe crossing that
    equilibrium at z ~ 0.10); LCDM measures 0.685. Those are 10% apart, which is almost
    certainly finer than this toy can resolve — the question being asked is not "is 0.618
    better than 0.685" but "does the framework's number hold the web open at all."
    """

    h0: float = 0.02
    omega_lambda: float = 1.0 / PHI
    a: float = 1.0

    @property
    def omega_m(self) -> float:
        return 1.0 - self.omega_lambda

    def hubble(self) -> float:
        return self.h0 * math.sqrt(max(0.0, self.omega_m * self.a ** -3 + self.omega_lambda))

    def advance(self, dt: float) -> None:
        self.a += self.a * self.hubble() * dt


@dataclass
class ParticleConfig:
    n: int = 4000
    box: float = 120.0              # COMOVING box; physical size is box * a
    dt: float = 0.05
    r0: float = 5.0                 # PHYSICAL interaction range — the one explicit length
    g: float = 0.8                  # gravity strength
    sec_balance: float = 0.6        # entropy pressure strength
    memory_decay: float = 0.95      # entropy fade
    damping: float = 0.99
    # --- the speed guard ---------------------------------------------------------------
    # None (default): the guard is the Courant displacement limit cfl * r0 / dt_eff, derived
    # every tick from the same rule that sets dt, so by construction it can bind only on the
    # top 1% tail the rule ignores. A number is honoured as an explicit cap (earlier
    # experiments swept it) and reported like any other bound.
    #
    # It used to be 2.0, unconditionally, and that number was the equation of motion. On
    # exp_11's 3D config (2026-08-28) every particle sat at it from tick ~100; forces only get
    # large once structure forms, so the cap saturated precisely when the engine started
    # doing something. 2.0 is BELOW the physical speed scale of these forces -- free fall
    # sqrt(2 a r0) ~ 5, drag-terminal a dt/(1-damping) ~ 6-40 -- so no fixed value near it can
    # be a guard that never binds. Lifting it to 20 was measured (poc_08) to let kinetic
    # energy grow 242x: it relocates the blowup. The step is what was wrong, not the cap.
    max_speed: Optional[float] = None
    cfl: float = 0.2                # Courant number: max displacement per step, in units of r0
    dt_min: Optional[float] = None  # floor on the step; None -> dt / 20. Reported when it binds.
    seed: int = 42
    # The tick at which `damping`, `memory_decay` and the SEC growth coefficient are STATED.
    # They are applied as rates -- x ** (dt_eff / dt_ref) -- which is bit-identical at
    # dt_eff == dt_ref and the only way an adaptive step can leave the physics alone: applied
    # per tick, a halved step would double the drag and the forgetting per unit time, a repair
    # that "works" by adding dissipation. Never change this to retune a run; change dt.
    dt_ref: float = 0.05
    entropy_init: float = 0.0       # exp_09 seeds 0; exp_11 seeds 0.1 * rand
    dims: int = 2                   # 2 or 3. exp_31 Part A: the cascade 1/r profile requires
                                    # d_spatial = 3, and the web topology exp_11 targets
                                    # (filaments, SHEETS, nodes, voids) only exists in 3D.
    cosmology: Optional[Cosmology] = None   # None = static box, reproducing POC-07 exactly

    # --- initial conditions -----------------------------------------------------------
    # "lattice"   perturbed grid — UNCORRELATED. What exp_09 and exp_11 used, deliberately:
    #             they ask whether the force law ALONE builds structure from near-uniform.
    #             Isotropic collapse into clumps is the expected outcome of that setup in any
    #             framework, so a low connectivity reading there is not a result about DFT.
    # "zeldovich" displacement off a P(k) ~ k^ic_index field — CORRELATED. The standard
    #             cosmological setup, and a DIFFERENT question: what does the force law do
    #             with realistic correlated initial conditions.
    #
    # Measured, 5 seeds, 500 steps, 2D: correlated ICs raise percolation from 0.107 +/- 0.024
    # to ~0.18-0.20, a +2.1 to +3.0 sigma effect. **The spectral index does not matter** over
    # -1.727 to -3.0 — all three overlap within their spreads. The default is therefore
    # anchored on exp_12's own measured output slope (P(k) ~ k^-1.727, "SCALE-FREE",
    # cosmic similarity 0.849) rather than on a hand-picked number.
    ic: str = "lattice"
    ic_index: float = -1.727        # spectral index of the seed field; exp_12's measured slope
    ic_amplitude: float = 1.0       # displacement in units of the lattice spacing

    # --- emergent local time ----------------------------------------------------------
    # "global"    one dt for everything — the substrate as it was.
    # "potential" clock rate from REMAINING collapse budget: slow where mass has piled up.
    # "rate"      clock rate from the INSTANTANEOUS collapse rate: slow at infall fronts.
    time_mode: str = "global"
    time_kappa: float = 1.0         # how hard the drive bites
    time_viscosity: float = 0.3     # nu: neighbour diffusion of tau, the anti-collapse term
    time_floor: float = 0.05        # nothing freezes completely; keeps dt bounded away from 0

    # How tau's viscosity term finds its neighbours. THIS IS A PHYSICS CHOICE, not a detail.
    #   "ball"  every particle within r0. Neighbour count then tracks LOCAL DENSITY -- ~798 in
    #           a web node against 116 in a uniform control at the same n and box -- so "how
    #           much matter is nearby" and "how it is connected" are entangled in the operator
    #           itself. exp_03 used this and concluded time does not conduct along the web; it
    #           could not have detected conduction either way.
    #   "knn"   the k nearest, so degree is BOUNDED and a dense region earns no extra coupling
    #           for being dense. exp_06 measured the web's conduction this way and found
    #           filaments ARE conduction paths, +9.14 sigma, k-independent.
    time_coupling: str = "ball"
    time_k: int = 6


class ParticleOperator(Protocol):
    name: str
    def __call__(self, s: ParticleState, c: ParticleConfig) -> ParticleState: ...


# ======================================================================================
# Correlated initial conditions
# ======================================================================================

def zeldovich(q: "torch.Tensor", c: ParticleConfig, spacing: float, device):
    """Displace a lattice by the gradient of a power-spectrum potential.

    Build a Gaussian field with P(k) ~ k^ic_index, solve for the displacement potential
    (psi_k = delta_k / k^2), take its gradient, and move each particle from its lattice
    site q by that displacement. Velocity is set proportional to the displacement, which is
    the growing mode: the flow that produced the offset keeps going.

    The reason this matters is anisotropy. A correlated overdensity collapses along its
    shortest axis first -- a sheet -- then along the next, giving a filament, then a node.
    That ordering is the Zel'dovich picture, and it is why a web appears at all. An
    uncorrelated start has no preferred axes to order, so every overdensity collapses to a
    round clump and the result is speckle regardless of the force law.

    Returns (positions, velocities), both comoving.
    """
    d, res = c.dims, int(round(c.box / spacing))
    res = max(res, 8)
    torch.manual_seed(c.seed + 991)

    # A real-valued Gaussian field, then shape its spectrum.
    delta = torch.fft.fftn(torch.randn(*([res] * d), device=device, dtype=torch.float32))
    freqs = [torch.fft.fftfreq(res, d=spacing, device=device) * 2 * math.pi for _ in range(d)]
    kk = torch.meshgrid(*freqs, indexing="ij")
    k2 = sum(x ** 2 for x in kk)
    k = torch.sqrt(k2)
    k2 = torch.where(k2 > 0, k2, torch.ones_like(k2))            # DC: no displacement

    amp = torch.where(k > 0, k.clamp(min=1e-12) ** (c.ic_index / 2.0), torch.zeros_like(k))
    delta = delta * amp
    delta[(0,) * d] = 0

    # psi = -grad(phi) with phi_k = delta_k / k^2  ->  psi_k = -i k delta_k / k^2
    disp = []
    for ax in range(d):
        psi_k = -1j * kk[ax] * delta / k2
        disp.append(torch.fft.ifftn(psi_k).real)
    disp = torch.stack(disp, dim=-1)

    # Normalise so the typical displacement is ic_amplitude lattice spacings, then sample
    # at each particle's lattice site.
    rms = disp.pow(2).sum(-1).sqrt().mean().clamp(min=1e-12)
    disp = disp * (c.ic_amplitude * spacing / rms)

    idx = ((q / spacing).long() % res)
    flat = idx[:, 0]
    for ax in range(1, d):
        flat = flat * res + idx[:, ax]
    dq = disp.reshape(-1, d)[flat]

    return (q + dq) % c.box, dq * 0.1


# ======================================================================================
# Geometry
# ======================================================================================

def pairwise(s: ParticleState, a: float = 1.0):
    """Minimum-image separations. Returns PHYSICAL distance and comoving direction.

    Positions are comoving, so a physical separation is `a` times the comoving one. Forces
    are evaluated at the physical distance — which is what makes expansion bite: as `a`
    grows, a fixed physical interaction range `r0` covers less and less comoving volume, and
    structure freezes out. With `a = 1` throughout this reduces exactly to the static case.

    Returns `(r_physical, d_comoving, r_comoving)`. BOTH radii are needed and confusing them
    is a silent scaling error: the force MAGNITUDE takes the physical distance, but the unit
    direction vector must be `d / r_comoving`, since `d` is comoving. Normalising a comoving
    displacement by a physical distance yields a vector of length 1/a, which would dilute
    gravity by an extra factor of `a` — indistinguishable in the output from expansion doing
    its job, and therefore fatal to this experiment.
    """
    d = s.pos.unsqueeze(1) - s.pos.unsqueeze(0)          # (N, N, 2) comoving
    d = d - s.box * torch.round(d / s.box)
    r_com = torch.sqrt((d ** 2).sum(-1) + 1e-8)
    r_com.fill_diagonal_(float("inf"))
    return r_com * a, d, r_com


def gravity_potential_table(g: float, r0: float, n: int = 65536):
    """U(r) for the gravity kernel, tabulated on [0, 3 r0]: U(r) = -int_r^{3 r0} g e^{-s/r0}/(s+0.1) ds.

    The force is F(r) = g e^{-r/r0}/(r+0.1) toward the neighbour, cut at 3 r0. This is the pair
    potential whose gradient it is, referenced to zero at the cutoff so u_ij is continuous there
    (the force is what jumps at the cutoff, and that jump is already the substrate's). Closed
    form for the test: U(r) = -g e^{0.1/r0} [E1((r+0.1)/r0) - E1((3 r0+0.1)/r0)]. No second
    gravity is introduced -- one kernel, one reduction. Returns numpy (r_grid, U_grid).
    n = 65536: the kernel is steepest at r = 0 (scale 0.1), and a 4096-point trapezoid misses the
    closed form there by 1e-4; at 65536 the table matches E1 to < 1e-6 everywhere (tested).
    """
    import numpy as np
    r = np.linspace(0.0, 3.0 * r0, n)
    f = g * np.exp(-r / r0) / (r + 0.1)
    # cumulative trapezoid from the cutoff inward: U(r) = -int_r^{R} f
    seg = 0.5 * (f[1:] + f[:-1]) * np.diff(r)
    tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    return r, -tail


def _interp_potential(r: torch.Tensor, within: torch.Tensor, table) -> torch.Tensor:
    """Linear lookup of U(r) on the table for pairs `within`; 0 elsewhere (and on the diagonal)."""
    r_grid, U_grid = table
    n = len(r_grid); span = float(r_grid[-1])
    Ug = torch.as_tensor(U_grid, dtype=r.dtype, device=r.device)
    r_safe = torch.where(within, r, torch.zeros_like(r))
    x = r_safe / span * (n - 1)
    idx = x.floor().clamp(0, n - 2).long()
    frac = (x - idx.to(r.dtype)).clamp(0.0, 1.0)
    u = Ug[idx] * (1.0 - frac) + Ug[idx + 1] * frac
    return torch.where(within, u, torch.zeros_like(u))


# ======================================================================================
# Operators
# ======================================================================================

class LocalGravity:
    """Finite-range attraction: F = G m_i m_j exp(-r/r0) / r, cut at 3 r0.

    Exponential rather than 1/r^2 — exp_09's result is that LOCAL gravity is sufficient for
    cosmic web topology and the Newtonian form is not required. The exponent is not asserted
    here; the law detector is asked afterwards what it measures.
    """

    name = "local_gravity"

    def __init__(self):
        self._table = None
        self._table_key = None

    def _potential_table(self, c: ParticleConfig):
        key = (float(c.g), float(c.r0))
        if self._table_key != key:
            self._table = gravity_potential_table(c.g, c.r0)
            self._table_key = key
        return self._table

    @torch.no_grad()
    def __call__(self, s: ParticleState, c: ParticleConfig) -> ParticleState:
        a = c.cosmology.a if c.cosmology else 1.0
        r, d, r_com = pairwise(s, a)
        within = r < 3.0 * c.r0
        mm = s.mass.unsqueeze(1) * s.mass.unsqueeze(0)
        mag = torch.where(within, c.g * mm * torch.exp(-r / c.r0) / (r + 0.1),
                          torch.zeros_like(r))
        # the same kernel, integrated: per-particle interaction energy (pairs counted from both ends)
        u = mm * _interp_potential(r, within, self._potential_table(c))
        potential_i = u.sum(dim=1)
        # d[i,j] = pos_i - pos_j points AWAY from j, so an attractive force needs -unit.
        # The first version used +unit and made gravity repulsive: the cloud expanded to
        # uniform, damping killed the motion, and every metric froze at t=100 — void 0.756
        # and cv 1.772 unchanged through t=600. exp_09 has the same sign with a comment
        # claiming it points toward the other particle; worth flagging there.
        unit = d / (r_com.unsqueeze(-1) + 1e-6)
        force = -(mag.unsqueeze(-1) * unit).sum(dim=1)         # toward neighbours
        m = dict(s.metrics)
        m["gravity_force_mean"] = force.norm(dim=-1).mean().item()
        m["potential_int"] = 0.5 * potential_i.sum().item()
        acc = force / s.mass.unsqueeze(-1)
        return s.replace(acc=acc if s.acc is None else s.acc + acc, acc_gravity=acc,
                         potential_i=potential_i, metrics=m)


class SECPressure:
    """Entropy pressure — the counter-force that opens voids.

    Range 2 r0 against gravity's 3 r0. Two competing interactions at *different* ranges is
    what selects a scale; a single monotone attraction only concentrates.

    **Pair law (2026-09-05): repulsion with magnitude sec_balance * (S_i + S_j)/2 * exp(-r/r0),
    always pushing the pair apart.** The rule inherited from exp_09 was sec * (S_i - S_j) along
    the unit vector from j to i. Under i <-> j both the difference and the direction flip, so
    the force on j from i was the SAME vector as the force on i from j: the antisymmetric part
    of the pair interaction was identically zero and the whole term was self-propulsion --
    every pair injected net momentum 2F (checked: F_i = F_j = -2.195 x for S = 5 and 1, six
    apart). There is no sign fix for that; the choice of a third-law pair law is a physics
    choice, and Peter chose the pressure form: strength set by the pair's mean entropy, so a
    dense hot region pushes outward from its interior. Third law exact; momentum is conserved
    by construction and tested (tests/v4/test_pressure_momentum.py). Every result before this
    commit -- POC-07/08/09/10, exp_04 -- was measured on the momentum-injecting rule.
    """

    name = "sec_pressure"

    @torch.no_grad()
    def __call__(self, s: ParticleState, c: ParticleConfig) -> ParticleState:
        a = c.cosmology.a if c.cosmology else 1.0
        r, d, r_com = pairwise(s, a)
        within = r < 2.0 * c.r0
        s_pair = 0.5 * (s.entropy.unsqueeze(1) + s.entropy.unsqueeze(0))   # symmetric in (i, j)
        mag = torch.where(within, c.sec_balance * s_pair * torch.exp(-r / c.r0),
                          torch.zeros_like(r))
        unit = d / (r_com.unsqueeze(-1) + 1e-6)                # from j to i: pushes i away
        press = (mag.unsqueeze(-1) * unit).sum(dim=1)
        m = dict(s.metrics)
        m["sec_pressure_mean"] = press.norm(dim=-1).mean().item()
        acc = press / s.mass.unsqueeze(-1)
        return s.replace(acc=acc if s.acc is None else s.acc + acc, acc_pressure=acc, metrics=m)


class SECUpdate:
    """Local entropy from local density. Dense regions accumulate; everyone forgets.

    This is the memory channel: entropy is a record of having been crowded, and it decays.
    Without the decay the pressure never releases and structure freezes.

    **The decay applies to every particle every tick.** Until 2026-09-05 it applied only on
    the NON-dense branch (`where(dense, entropy + growth, entropy * memory_decay)`), so a
    particle flagged dense never forgot. Once `dense_fraction` reached ~0.97 (exp_11's 3D
    config, tick ~200) 97% of particles were on the growth branch and the memory could not
    release: entropy ran 100 -> 1728 -> 4419 monotonically, the pressure it drives followed,
    and the speed cap locked. The docstring above described the intended physics; the code
    did not implement it.

    With unconditional release the entropy of any particle is bounded by
        0.1 * (n - expected) / (1 - memory_decay)
    -- growth at most 0.1 * (n - expected) per tick against geometric decay -- with no cap
    and no new knob. A cap is deliberately not added: with release present it is inert
    (dawn-field-theory/experiments/spikes/postsymbolic_selection).

    Growth and decay are RATES against `dt_ref` (see ParticleConfig): at the base step the
    non-dense branch is bit-identical to the old code.
    """

    name = "sec_update"

    @torch.no_grad()
    def __call__(self, s: ParticleState, c: ParticleConfig) -> ParticleState:
        a = c.cosmology.a if c.cosmology else 1.0
        r, _, _ = pairwise(s, a)
        local = (r < c.r0).sum(dim=1).float()
        # Volume of a d-ball, not a disc: pi r^2 in 2D but (4/3) pi r^3 in 3D. Using the 2D
        # form in 3D makes the density trigger wrong by a factor of order unity, so entropy
        # would accumulate at the wrong places.
        d = s.pos.shape[1]
        v_ball = (math.pi ** (d / 2) / math.gamma(d / 2 + 1)) * c.r0 ** d
        expected = float(s.n) * v_ball / ((s.box * a) ** d)
        dense = local > 1.5 * expected
        s_dt = (s.dt_last if s.dt_last is not None else c.dt) / c.dt_ref
        growth = 0.1 * (local - expected) * s_dt
        ent = (s.entropy * (c.memory_decay ** s_dt)
               + torch.where(dense, growth, torch.zeros_like(growth)))
        m = dict(s.metrics)
        m["entropy_mean"] = ent.mean().item()
        m["entropy_max"] = ent.max().item()
        m["dense_fraction"] = dense.float().mean().item()
        return s.replace(entropy=ent, metrics=m)


class SECUpdateRelative:
    """exp_11's SEC rule, which is NOT exp_09's.

        entropy += sec_balance * (local_count - mean_count) / (mean_count + 1)

    Two differences from `SECUpdate`, and they are not cosmetic:

    * **No threshold.** exp_09 only accumulates where density exceeds 1.5x expected;
      exp_11 responds smoothly to deviation in both directions, so under-dense regions
      lose entropy rather than merely decaying.
    * **No decay.** exp_09 multiplies by `memory_decay` outside dense regions; exp_11 has
      no forgetting at all, only the clamp at zero.

    Kept as a separate operator rather than a flag because the two references genuinely ran
    different physics, and a replication that quietly used the wrong one would be worthless.
    exp_11 also seeds entropy at `0.1 * rand` rather than zero, which `ParticleEngine`
    handles via `entropy_init`.
    """

    name = "sec_update_relative"

    @torch.no_grad()
    def __call__(self, s: ParticleState, c: ParticleConfig) -> ParticleState:
        a = c.cosmology.a if c.cosmology else 1.0
        r, _, _ = pairwise(s, a)
        local = (r < c.r0).sum(dim=1).float()
        mean_count = local.mean()
        deviation = (local - mean_count) / (mean_count + 1.0)
        # Rate against dt_ref, so the literal exp_11 transcription is preserved at the base
        # step and does not acquire a dt-dependence when the Integrator shrinks it.
        s_dt = (s.dt_last if s.dt_last is not None else c.dt) / c.dt_ref
        ent = torch.clamp(s.entropy + c.sec_balance * deviation * s_dt, min=0.0)
        m = dict(s.metrics)
        m["entropy_mean"] = ent.mean().item()
        m["local_count_mean"] = mean_count.item()
        return s.replace(entropy=ent, metrics=m)


class LocalTime:
    """Emergent local time: a per-particle clock rate set by the local collapse budget.

    The premise is that a collapse event IS a tick, so the local framerate is set by how
    much collapse the neighbourhood can still do. Mass is spent potential, so a mass-dense
    region has little budget left and ticks slowly. **Gravitational time dilation then falls
    out of PAC bookkeeping rather than being inserted** — nothing here knows about GR.

    Two candidate sources, because the corpus does not fix the choice and they diverge in
    exactly the regime that matters (a region can be potential-rich but quiescent):

      "potential"  tau ~ 1/(1 + kappa*delta)          — REMAINING budget.
                   Slow wherever matter has piled up. Clocks track where collapse HAS gone.
      "rate"       tau ~ 1/(1 + kappa*|d delta/dt|)   — INSTANTANEOUS collapse rate.
                   Slow only where collapse is happening NOW, so a settled node ticks fast
                   again and the slow surfaces are the infall fronts.

    They predict different topologies, which is the point: under "potential" the slow-time
    set should trace the filaments and nodes; under "rate" it should trace their boundaries.
    Measure, do not assert.

    **tau is normalised to mean 1.** Local time is then a REDISTRIBUTION of a fixed global
    time budget, never a global speed-up, which is what keeps the ledger auditable: total
    elapsed proper time is conserved by construction and only its dispersion is physical.

    `time_viscosity` diffuses tau across neighbours — the nu*grad^2(tau) term. This is the
    "dispersed evenly enough not to collapse" condition made concrete: without it, tau
    gradients sharpen until neighbouring regions tick at wildly different rates and stop
    being causally coupled. Viscosity is what keeps the field well-posed, not a cosmetic
    smoother. `tau_dispersion` is reported every tick so that decoupling is visible rather
    than silent.
    """

    name = "local_time"

    @torch.no_grad()
    def __call__(self, s: ParticleState, c: ParticleConfig) -> ParticleState:
        if c.time_mode == "global":
            return s
        a = c.cosmology.a if c.cosmology else 1.0
        r, _, _ = pairwise(s, a)
        near = (r < c.r0)
        count = near.sum(dim=1).float()
        delta = count / count.mean().clamp(min=1e-9) - 1.0        # local overdensity

        if c.time_mode == "rate":
            prev = s.prev_delta if s.prev_delta is not None else delta
            drive = (delta - prev).abs() / max(c.dt, 1e-9)
        else:                                                      # "potential"
            drive = delta.clamp(min=0.0)

        tau = 1.0 / (1.0 + c.time_kappa * drive)

        # nu * grad^2(tau) — the anti-collapse stabiliser, and (exp_02) the ONLY channel by
        # which tau acquires non-local structure. Which neighbours it averages over is a
        # physics choice: see ParticleConfig.time_coupling.
        if c.time_viscosity > 0:
            if c.time_coupling == "knn":
                rr = r.clone()
                rr.fill_diagonal_(float("inf"))
                k = min(c.time_k, rr.shape[0] - 1)
                idx = rr.topk(k, dim=1, largest=False).indices
                adj = torch.zeros_like(near)
                adj.scatter_(1, idx, True)
                w = (adj | adj.T).float()
            else:
                w = near.float()
            neigh = (w @ tau) / w.sum(dim=1).clamp(min=1.0)
            tau = tau + c.time_viscosity * (neigh - tau)
            m0 = dict(s.metrics)
            m0["tau_coupling_degree"] = w.sum(dim=1).mean().item()

        tau = tau.clamp(min=c.time_floor)
        tau = tau / tau.mean().clamp(min=1e-9)                     # redistribution, not speed-up

        pt = (s.proper_time if s.proper_time is not None
              else torch.zeros_like(tau)) + tau * c.dt
        m = dict(m0) if c.time_viscosity > 0 else dict(s.metrics)
        m["tau_mean"] = tau.mean().item()
        m["tau_dispersion"] = (tau.std() / tau.mean().clamp(min=1e-9)).item()
        m["tau_min"] = tau.min().item()
        m["tau_max"] = tau.max().item()
        m["proper_time_spread"] = (pt.max() - pt.min()).item()
        return s.replace(tau=tau, proper_time=pt, prev_delta=delta, metrics=m)


class Integrator:
    """Kick, damp, guard, drift -- on ONE global step chosen from this tick's forces.

    The force operators accumulate acceleration into `state.acc`; this operator applies it.
    Owning the step here, rather than each force kicking with `config.dt`, is what lets the
    step be set from the forces that are about to act instead of from a fixed number.

    **The step.** A Courant rule on a ROBUST statistic of this tick's accelerations and
    velocities:

        a99 = |acc|.quantile(0.99)         v99 = (|vel| tau_i).quantile(0.99)
        dt  = min(c.dt, cfl sqrt(r0 / a99), cfl r0 / (v99 + a99 dt))
        dt  = max(dt, dt_min)              cap = cfl r0 / dt

    The first bound resolves the force's own length scale (a dt^2 <= cfl^2 r0); the second
    bounds displacement per step. p99 rather than max, so the 1% most extreme particles never
    set everyone's clock (the v3 field engine's TimeEmergence has that defect, documented in
    its own docstring), and the guard then binds on at most that tail BY CONSTRUCTION. On
    exp_11's quiescent phase (a99 ~ 8, v99 ~ 5) the rule leaves dt at c.dt: the step was never
    the problem for gravity alone. When SEC pressure detonates a clump (a99 ~ 120, v99 ~ 300)
    it drops to ~0.007 and the cap rises to ~300 with it, so the force law keeps being
    integrated where before it was discarded.

    **Rates, not per-tick constants.** `damping` is applied as damping ** (dt / dt_ref). At
    dt == dt_ref this is bit-identical to multiplying by `damping` once; at any other step it
    is the same drag per unit time. Without this an adaptive step would add dissipation
    whenever it shrank -- a repair that works by turning the physics off.

    **What is reported, every tick.** `dt_eff`, `dt_at_floor`, `cap_eff`, `at_cap_frac` (from
    the pre-clamp speed, tolerance as in the diagnostic), `speed_p99`, `speed_max`,
    `accel_p99`, `cfl_number` = max(v99 dt, a99 dt^2) / r0, and `sim_time`, the true elapsed
    global time. Read `at_cap_frac` beside any dynamical quantity; above a few percent with
    `dt` off its floor, the controller is wrong (spec R3/R5), not the physics.

    When `state.tau` is present each particle DRIFTS by its own dt_i = dt tau_i, as before;
    the kick is on the global dt (emergent local time in the kick is `.spec/challenges.md`
    C4.2, a physics decision, not hygiene). With `time_mode="global"` the substrate at base
    dt with no forces is bit-identical to the pre-2026-09-05 integrator.
    """

    name = "integrator"

    @torch.no_grad()
    def __call__(self, s: ParticleState, c: ParticleConfig) -> ParticleState:
        acc = s.acc if s.acc is not None else torch.zeros_like(s.vel)
        tau = None if s.tau is None else s.tau.unsqueeze(-1)

        # --- the step, from this tick's forces -------------------------------------------
        a_mag = acc.norm(dim=-1)
        v_mag = s.vel.norm(dim=-1) if tau is None else (s.vel * tau).norm(dim=-1)
        a99 = torch.quantile(a_mag, 0.99).item()
        v99 = torch.quantile(v_mag, 0.99).item()
        dt = c.dt
        if a99 > 0.0:
            dt = min(dt, c.cfl * math.sqrt(c.r0 / a99))
        denom = v99 + a99 * dt
        if denom > 0.0:
            dt = min(dt, c.cfl * c.r0 / denom)
        dt_min = c.dt_min if c.dt_min is not None else c.dt / 20.0
        at_floor = dt <= dt_min
        dt = max(dt, dt_min)
        cap = c.max_speed if c.max_speed is not None else c.cfl * c.r0 / dt

        # --- kick, then damp as a rate ------------------------------------------------------
        v0 = s.vel
        v1 = v0 + acc * dt
        vel = v1 * (c.damping ** (dt / c.dt_ref))
        # Exact discrete partition of the kick's kinetic change by force: with a = sum_X a_X,
        #   dKE_kick = sum_X [ m v0.a_X dt + 1/2 m a_X.a dt^2 ]   (identity, not an approximation)
        mvec = s.mass.unsqueeze(-1)
        a_g = s.acc_gravity if s.acc_gravity is not None else torch.zeros_like(acc)
        a_p = s.acc_pressure if s.acc_pressure is not None else torch.zeros_like(acc)
        def _work(a_x):
            return ((mvec * v0 * a_x).sum() * dt + 0.5 * (mvec * a_x * acc).sum() * dt * dt).item()
        work_g, work_p = _work(a_g), _work(a_p)
        ke1 = (0.5 * s.mass * (v1 ** 2).sum(-1)).sum().item()
        ke2 = (0.5 * s.mass * (vel ** 2).sum(-1)).sum().item()
        loss_drag = ke1 - ke2
        impulse_p = (mvec * a_p).sum(0) * dt
        if c.cosmology is not None:
            # Standard comoving form: peculiar velocities decay as dv/dt = -2 H v, and
            # comoving displacement is v/a. This is what "expansion holds the web open"
            # actually means mechanically -- infall is fought by the drag and by the
            # separation growing underneath it.
            H = c.cosmology.hubble()
            vel = vel * (1.0 - 2.0 * H * dt)

        # --- guard: measured before it acts, then applied ----------------------------------
        speed = vel.norm(dim=-1, keepdim=True)
        m = dict(s.metrics)
        m["at_cap_frac"] = (speed >= cap * 0.999).float().mean().item()
        m["speed_p99"] = torch.quantile(speed.flatten(), 0.99).item()
        m["speed_max"] = speed.max().item()
        m["cap_eff"] = float(cap)
        m["dt_eff"] = float(dt)
        m["dt_at_floor"] = bool(at_floor)
        m["accel_p99"] = a99
        m["cfl_number"] = max(v99 * dt, a99 * dt * dt) / c.r0
        m["sim_time"] = float(s.metrics.get("sim_time", 0.0)) + dt
        vel = torch.where(speed > cap, vel * cap / speed, vel)
        ke3 = (0.5 * s.mass * (vel ** 2).sum(-1)).sum().item()
        loss_guard = ke2 - ke3
        m["work_gravity"], m["work_pressure"] = work_g, work_p
        m["loss_drag"], m["loss_guard"] = loss_drag, loss_guard
        for k, ax in enumerate("xyz"[: s.pos.shape[1]]):
            m[f"impulse_pressure_{ax}"] = impulse_p[k].item()
        for key, val in (("work_gravity", work_g), ("work_pressure", work_p),
                         ("loss_drag", loss_drag), ("loss_guard", loss_guard)):
            m[key + "_cum"] = float(m.get(key + "_cum", 0.0)) + val

        # --- drift ------------------------------------------------------------------------
        a = c.cosmology.a if c.cosmology else 1.0
        dt_i = dt if tau is None else dt * tau
        pos = (s.pos + vel * dt_i / a) % s.box
        return s.replace(pos=pos, vel=vel, acc=None, acc_gravity=None, acc_pressure=None,
                         dt_last=dt, metrics=m)


class PACLedger:
    """Read-only audit. Records the ledger rather than correcting it.

    The field engine enforces `E + I + M` with an explicit global correction every tick, so
    its one conserved quantity is enforced rather than observed. Here nothing is corrected —
    whatever conserves, conserves, and the law detector is the judge.
    """

    name = "pac_ledger"

    @torch.no_grad()
    def __call__(self, s: ParticleState, c: ParticleConfig) -> ParticleState:
        m = dict(s.metrics)
        ke_prev = s.metrics.get("kinetic_int")
        m["mass_total"] = s.mass.sum().item()
        m["kinetic"] = (0.5 * s.mass * (s.vel ** 2).sum(-1)).sum().item()
        m["entropy_total"] = s.entropy.sum().item()
        # --- the interacting set's energy budget (all particles until severance exists) -------
        m["n_alive"] = int(s.n)
        m["kinetic_int"] = m["kinetic"]
        m["potential_int"] = (0.5 * s.potential_i.sum().item() if s.potential_i is not None
                              else float(s.metrics.get("potential_int", 0.0)))
        m["total_int"] = m["kinetic_int"] + m["potential_int"]
        m["e_int"] = m["total_int"] / max(m["n_alive"], 1)
        # closure: this tick's kinetic change must be exactly the work minus the losses.
        # A sign or partition error anywhere upstream shows up here as O(1), not as drift.
        if ke_prev is not None:
            budget = (m.get("work_gravity", 0.0) + m.get("work_pressure", 0.0)
                      - m.get("loss_drag", 0.0) - m.get("loss_guard", 0.0)
                      - m.get("loss_landauer", 0.0) - m.get("loss_severance_ke", 0.0))
            m["closure_residual"] = abs((m["kinetic_int"] - ke_prev) - budget) / max(abs(ke_prev), 1.0)
        else:
            m["closure_residual"] = 0.0
        # One key per spatial axis. This recorded x and y only, so on every 3D run the z
        # component was silently missing from the conservation ledger.
        p = (s.mass.unsqueeze(-1) * s.vel).sum(0)
        for k, ax in enumerate("xyz"[: s.pos.shape[1]]):
            m[f"momentum_{ax}"] = p[k].item()
        return s.replace(metrics=m)


CANONICAL: list[Callable[[], ParticleOperator]] = [
    SECUpdate, LocalGravity, SECPressure, Integrator, PACLedger,
]

# Same forces, with emergent local time in front of the integrator so the clock rate is set
# from the CURRENT configuration before anything moves. Inert unless time_mode != "global".
CANONICAL_TIME: list[Callable[[], ParticleOperator]] = [
    SECUpdate, LocalGravity, SECPressure, LocalTime, Integrator, PACLedger,
]

# exp_11's 3D pipeline: same forces, different SEC rule. Its ordering also differs — exp_11
# updates entropy AFTER moving, where exp_09 updates before.
EXP11 = [LocalGravity, SECPressure, Integrator, SECUpdateRelative, PACLedger]
EXP11_TIME = [LocalGravity, SECPressure, LocalTime, Integrator, SECUpdateRelative, PACLedger]


# ======================================================================================
# Engine
# ======================================================================================

class ParticleEngine:
    # Above this fraction of particles at the cap, the substrate is integrating a direction
    # field rather than the force law (see Integrator). The engine annotates always and
    # warns ONCE; it never asserts, because an assertion kills an exploratory run that may
    # be the very run that shows where the bound binds. Tests assert; the engine reports.
    AT_CAP_WARN = 0.02

    def __init__(self, config: ParticleConfig | None = None,
                 pipeline: Optional[list] = None, device=None):
        self.config = config or ParticleConfig()
        self.pipeline = [op() for op in (pipeline or CANONICAL)]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = self._init()
        self.tick_count = 0
        # Running record of every numerical bound, so a run can be read for "did anything
        # bind" without re-running it. Written every tick from state.metrics.
        self.bounds = {"at_cap_frac_max": 0.0, "ticks_at_cap_gt_1pct": 0,
                       "first_tick_at_cap_gt_1pct": None, "ticks_at_dt_floor": 0,
                       "dt_eff_min": None}
        self._warned_cap = False

    def _init(self) -> ParticleState:
        c = self.config
        torch.manual_seed(c.seed)
        d = c.dims
        per = int(math.ceil(c.n ** (1.0 / d)))
        sp = c.box / per
        g = torch.arange(per, device=self.device, dtype=torch.float32) * sp + sp / 2
        grids = torch.meshgrid(*([g] * d), indexing="ij")
        pos = torch.stack([x.flatten()[:c.n] for x in grids], dim=1)
        if pos.shape[0] < c.n:       # a perfect d-th root rarely divides n exactly
            pad = c.n - pos.shape[0]
            pos = torch.cat([pos, torch.rand(pad, d, device=self.device) * c.box], dim=0)
        if c.ic == "zeldovich":
            pos, vel = zeldovich(pos, c, sp, self.device)
        else:
            pos = (pos + torch.randn_like(pos) * sp * 0.1) % c.box
            vel = torch.zeros_like(pos)
        return ParticleState(
            pos=pos,
            vel=vel,
            mass=1.0 + 0.1 * torch.randn(c.n, device=self.device),
            entropy=(c.entropy_init * torch.rand(c.n, device=self.device)
                     if c.entropy_init else torch.zeros(c.n, device=self.device)),
            box=c.box,
        )

    def tick(self) -> ParticleState:
        s = self.state
        for op in self.pipeline:
            s = op(s, self.config)
        if self.config.cosmology is not None:
            self.config.cosmology.advance(s.dt_last if s.dt_last is not None else self.config.dt)
        self.state = s
        self.tick_count += 1
        self._record_bounds(s)
        return s

    def _record_bounds(self, s: ParticleState) -> None:
        frac = s.metrics.get("at_cap_frac")
        if frac is None:
            return
        b = self.bounds
        b["at_cap_frac_max"] = max(b["at_cap_frac_max"], frac)
        if s.metrics.get("dt_at_floor"):
            b["ticks_at_dt_floor"] += 1
        d = s.metrics.get("dt_eff")
        if d is not None:
            b["dt_eff_min"] = d if b["dt_eff_min"] is None else min(b["dt_eff_min"], d)
        if frac > 0.01:
            b["ticks_at_cap_gt_1pct"] += 1
            if b["first_tick_at_cap_gt_1pct"] is None:
                b["first_tick_at_cap_gt_1pct"] = self.tick_count
        if frac > self.AT_CAP_WARN and not self._warned_cap:
            self._warned_cap = True
            warnings.warn(
                f"speed guard binding: {100 * frac:.1f}% of particles at cap="
                f"{s.metrics.get('cap_eff', self.config.max_speed)} on tick {self.tick_count}; the force law is not "
                f"being integrated for them (see Integrator). Reported once per engine; "
                f"engine.bounds keeps the running record.",
                RuntimeWarning, stacklevel=2)

    def field_of(self, values: torch.Tensor, res: int,
                 weight: Optional[torch.Tensor] = None):
        """Bin a per-particle quantity to a grid as a weighted MEAN (NaN where empty).

        `density_field` sums over particles, which is right for density. An intensive
        quantity -- a clock rate, an entropy, a temperature -- has to be AVERAGED instead:
        summing it would just re-measure density wearing a different label, which is the kind
        of quiet tautology this directory has paid for before.
        """
        s = self.state
        d = s.pos.shape[1]
        idx = (s.pos / s.box * res).long().clamp(0, res - 1)
        flat = idx[:, 0]
        for ax in range(1, d):
            flat = flat * res + idx[:, ax]
        w = torch.ones_like(values) if weight is None else weight
        num = torch.zeros(res ** d, device=s.pos.device).scatter_add_(0, flat, values * w)
        den = torch.zeros(res ** d, device=s.pos.device).scatter_add_(0, flat, w)
        out = torch.where(den > 0, num / den.clamp(min=1e-12),
                          torch.full_like(num, float("nan")))
        return out.reshape(*([res] * d))

    def density_field(self, res: int = 128) -> torch.Tensor:
        """Bin to a d-dimensional grid so the instruments in structure.py apply unchanged.

        A 3D grid at res=128 is 2M cells and the flood fill behind `percolation` walks all of
        them, so callers should drop to res~64 in 3D. The binning itself is res-agnostic.
        """
        s = self.state
        d = s.pos.shape[1]
        idx = [(s.pos[:, k] / s.box * res).long().clamp(0, res - 1) for k in range(d)]
        flat = idx[0]
        for k in range(1, d):
            flat = flat * res + idx[k]
        f = torch.zeros(res ** d, device=s.device)
        f.scatter_add_(0, flat, s.mass)
        return f.view(*([res] * d))
