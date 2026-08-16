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
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Protocol

import torch

PHI = (1 + math.sqrt(5)) / 2


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
    max_speed: float = 2.0
    seed: int = 42
    entropy_init: float = 0.0       # exp_09 seeds 0; exp_11 seeds 0.1 * rand
    dims: int = 2                   # 2 or 3. exp_31 Part A: the cascade 1/r profile requires
                                    # d_spatial = 3, and the web topology exp_11 targets
                                    # (filaments, SHEETS, nodes, voids) only exists in 3D.
    cosmology: Optional[Cosmology] = None   # None = static box, reproducing POC-07 exactly


class ParticleOperator(Protocol):
    name: str
    def __call__(self, s: ParticleState, c: ParticleConfig) -> ParticleState: ...


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

    @torch.no_grad()
    def __call__(self, s: ParticleState, c: ParticleConfig) -> ParticleState:
        a = c.cosmology.a if c.cosmology else 1.0
        r, d, r_com = pairwise(s, a)
        within = r < 3.0 * c.r0
        mm = s.mass.unsqueeze(1) * s.mass.unsqueeze(0)
        mag = torch.where(within, c.g * mm * torch.exp(-r / c.r0) / (r + 0.1),
                          torch.zeros_like(r))
        # d[i,j] = pos_i - pos_j points AWAY from j, so an attractive force needs -unit.
        # The first version used +unit and made gravity repulsive: the cloud expanded to
        # uniform, damping killed the motion, and every metric froze at t=100 — void 0.756
        # and cv 1.772 unchanged through t=600. exp_09 has the same sign with a comment
        # claiming it points toward the other particle; worth flagging there.
        unit = d / (r_com.unsqueeze(-1) + 1e-6)
        force = -(mag.unsqueeze(-1) * unit).sum(dim=1)         # toward neighbours
        m = dict(s.metrics)
        m["gravity_force_mean"] = force.norm(dim=-1).mean().item()
        return s.replace(vel=s.vel + force * c.dt / s.mass.unsqueeze(-1), metrics=m)


class SECPressure:
    """Entropy-gradient repulsion — the counter-force that opens voids.

    Range 2 r0 against gravity's 3 r0. Two competing interactions at *different* ranges is
    what selects a scale; a single monotone attraction only concentrates.
    """

    name = "sec_pressure"

    @torch.no_grad()
    def __call__(self, s: ParticleState, c: ParticleConfig) -> ParticleState:
        a = c.cosmology.a if c.cosmology else 1.0
        r, d, r_com = pairwise(s, a)
        within = r < 2.0 * c.r0
        de = s.entropy.unsqueeze(1) - s.entropy.unsqueeze(0)
        mag = torch.where(within, c.sec_balance * de * torch.exp(-r / c.r0),
                          torch.zeros_like(r))
        unit = d / (r_com.unsqueeze(-1) + 1e-6)                # away from neighbours
        press = (mag.unsqueeze(-1) * unit).sum(dim=1)
        m = dict(s.metrics)
        m["sec_pressure_mean"] = press.norm(dim=-1).mean().item()
        return s.replace(vel=s.vel + press * c.dt / s.mass.unsqueeze(-1), metrics=m)


class SECUpdate:
    """Local entropy from local density. Dense regions accumulate; sparse ones forget.

    This is the memory channel: entropy is a record of having been crowded, and it decays.
    Without the decay the pressure never releases and structure freezes.
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
        ent = torch.where(dense, s.entropy + 0.1 * (local - expected),
                          s.entropy * c.memory_decay)
        m = dict(s.metrics)
        m["entropy_mean"] = ent.mean().item()
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
        ent = torch.clamp(s.entropy + c.sec_balance * deviation, min=0.0)
        m = dict(s.metrics)
        m["entropy_mean"] = ent.mean().item()
        m["local_count_mean"] = mean_count.item()
        return s.replace(entropy=ent, metrics=m)


class Integrator:
    """Damped drift with a speed cap, on a periodic box."""

    name = "integrator"

    @torch.no_grad()
    def __call__(self, s: ParticleState, c: ParticleConfig) -> ParticleState:
        vel = s.vel * c.damping
        if c.cosmology is not None:
            # Standard comoving form: peculiar velocities decay as dv/dt = -2 H v, and
            # comoving displacement is v/a. This is what "expansion holds the web open"
            # actually means mechanically — infall is fought by the drag and by the
            # separation growing underneath it.
            H = c.cosmology.hubble()
            vel = vel * (1.0 - 2.0 * H * c.dt)
        speed = vel.norm(dim=-1, keepdim=True)
        vel = torch.where(speed > c.max_speed, vel * c.max_speed / speed, vel)
        a = c.cosmology.a if c.cosmology else 1.0
        pos = (s.pos + vel * c.dt / a) % s.box
        return s.replace(pos=pos, vel=vel)


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
        m["mass_total"] = s.mass.sum().item()
        m["kinetic"] = (0.5 * s.mass * (s.vel ** 2).sum(-1)).sum().item()
        m["entropy_total"] = s.entropy.sum().item()
        m["momentum_x"] = (s.mass.unsqueeze(-1) * s.vel).sum(0)[0].item()
        m["momentum_y"] = (s.mass.unsqueeze(-1) * s.vel).sum(0)[1].item()
        return s.replace(metrics=m)


CANONICAL: list[Callable[[], ParticleOperator]] = [
    SECUpdate, LocalGravity, SECPressure, Integrator, PACLedger,
]

# exp_11's 3D pipeline: same forces, different SEC rule. Its ordering also differs — exp_11
# updates entropy AFTER moving, where exp_09 updates before.
EXP11 = [LocalGravity, SECPressure, Integrator, SECUpdateRelative, PACLedger]


# ======================================================================================
# Engine
# ======================================================================================

class ParticleEngine:
    def __init__(self, config: ParticleConfig | None = None,
                 pipeline: Optional[list] = None, device=None):
        self.config = config or ParticleConfig()
        self.pipeline = [op() for op in (pipeline or CANONICAL)]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = self._init()
        self.tick_count = 0

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
        pos = (pos + torch.randn_like(pos) * sp * 0.1) % c.box
        return ParticleState(
            pos=pos,
            vel=torch.zeros_like(pos),
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
            self.config.cosmology.advance(self.config.dt)
        self.state = s
        self.tick_count += 1
        return s

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
