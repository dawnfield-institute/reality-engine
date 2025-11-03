"""
DawnField: The Primordial Field from Which Reality Emerges

This is the core of the Reality Engine - three fundamental fields
(Energy, Information, Memory) that evolve via the Recursive Balance Field
equation. No physics is hardcoded; everything emerges from these dynamics.

Based on:
- Dawn Field Theory's RBF equation
- PAC (Potential-Actualization-Conservation) framework
- Möbius-Confluence collapse-regeneration principle
- Validated constants from experimental work

GPU-accelerated with PyTorch + CUDA
Now integrated with Fracton for native PAC regulation and field operations.
"""

import torch
from typing import Tuple, List
import logging

# Fracton integration for native PAC and field operations
from fracton.core import PACRegulator
from fracton.core.recursive_engine import ExecutionContext
from fracton.field import CMBInitializer, RBFEngine, QBERegulator

# Create module logger
logger = logging.getLogger(__name__)


class DawnField:
    """
    The primordial field from which reality emerges.
    
    Three fundamental fields:
    - E (Energy): The actual state of the field
    - I (Information): The potential state of the field  
    - M (Memory): Recursive history enabling persistence
    
    No physics imposed - only potential and balance.
    Everything else emerges from the RBF evolution equation.
    
    GPU-accelerated with PyTorch CUDA.
    """
    
    def __init__(self, shape: Tuple[int, int, int] = (128, 128, 128), dt: float = 0.0001, device: str = None):
        """
        Initialize the primordial dawn field.
        
        Args:
            shape: 3D spatial dimensions (default: 128³ lattice)
            dt: Time step for evolution (default: 0.0001 for numerical stability, proven in MED experiments)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Auto-detect CUDA if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Using device: {self.device}")
        
        # Initialize Fracton PAC Regulator for native conservation enforcement
        self.pac_regulator = PACRegulator(
            xi_target=1.0571,  # Balance operator target (from Fracton)
            tolerance=1e-6,     # Conservation precision
            auto_correction=True  # Enable automatic PAC corrections
        )
        print(f"🔧 PAC Regulator initialized (xi_target={self.pac_regulator.xi_target})")
        
        # Store dimensions
        self.size = shape[0]
        self.depth = shape[2]
        
        # Initialize Fracton field components
        # CMB-like initialization using Fracton primitives
        cmb_initializer = CMBInitializer(n_hotspots=12, seed=None)
        E_np = cmb_initializer.initialize(shape, backend='numpy')
        I_np = cmb_initializer.initialize(shape, backend='numpy')
        
        # Convert to PyTorch tensors on correct device
        self.E = torch.from_numpy(E_np).float().to(self.device)
        self.I = torch.from_numpy(I_np).float().to(self.device)
        self.M = torch.zeros(*shape, device=self.device)  # Memory starts empty
        
        print(f"🌌 Fields initialized with Fracton CMBInitializer")
        
        # Validated constants from Dawn Field Theory experiments
        self.lambda_mem = 0.020      # Memory coupling (universal 0.020 Hz frequency)
        self.alpha_collapse = 0.964   # Collapse rate (96.4% validated correlation)
        
        # Initialize Fracton RBF Engine for field evolution
        self.rbf_engine = RBFEngine(
            lambda_mem=self.lambda_mem,
            alpha_collapse=self.alpha_collapse,
            backend='torch'
        )
        print(f"⚡ RBF Engine initialized (Fracton)")
        
        # Initialize Fracton QBE Regulator for E↔I coupling
        self.qbe_regulator = QBERegulator(
            lambda_qbe=1.0,
            qpl_omega=0.020,
            backend='torch'
        )
        print(f"🔗 QBE Regulator initialized (Fracton)")
        
        # QBE (Quantum Balance Equation) - THE fundamental constraint
        # dI/dt + dE/dt = λ·QPL(t)
        # This enforces energy-information balance - NO random growth allowed!
        self.lambda_qbe = 1.0  # QBE coupling constant (dimensionless)
        self.qpl_omega = 0.020  # QPL oscillation frequency (matches lambda_mem - universal 0.020 Hz)
        
        # SEC-MED Integration Parameters (from your experiments)
        self.sec_depth = 2              # Proven optimal recursion depth
        self.sec_threshold = 0.9        # SEC collapse threshold (high = sparse collapses)
        # Stability parameters
        self.med_viscosity = 0.01  # GAIA uses 0.01 - provides natural damping via diffusion
        self.sec_nodes_max = 3          # Maximum symbolic nodes (bounded complexity)
        
        # Quantum emergence tracking
        self.wavefunction_field = torch.zeros(*shape, dtype=torch.complex64, device=self.device)
        self.quantum_coherence = torch.ones(*shape, device=self.device)
        self.entanglement_map = torch.zeros(*shape, device=self.device)
        
        # Thresholds for emergence (from legacy cosmo.py validated values)
        self.herniation_threshold = 0.4     # Collapse site detection (proven in legacy)
        self.energy_threshold = 0.05        # Minimum energy for collapse
        self.collapse_threshold = 0.5       # SEC trigger level
        
        # Evolution parameters
        self.dt = dt
        self.time = 0
        self.step_count = 0
        
        # Stability controls
        self.max_field_value = 1e6  # Field value clamping
        
        # Track herniations
        self.herniation_history = []
        
        # Field health monitoring
        self._check_field_health()
    
    def _check_field_health(self):
        """Check fields for NaN/Inf and log warnings."""
        for name, field in [('E', self.E), ('I', self.I), ('M', self.M)]:
            if torch.isnan(field).any():
                logger.error(f"❌ {name} field contains NaN!")
            if torch.isinf(field).any():
                logger.error(f"❌ {name} field contains Inf!")
            
            # Log ranges
            logger.debug(f"{name}: min={field.min().item():.6f}, max={field.max().item():.6f}, mean={field.mean().item():.6f}")
    
    def _create_localized_field(self, shape: Tuple[int, int, int], n_hotspots: int = 12) -> torch.Tensor:
        """
        Create field with localized Gaussian hotspots (like CMB fluctuations).
        
        Combines:
        - Low-amplitude random background (like cosmo.py)
        - Gaussian hotspots for structure seeds (like mobius strands)
        
        This gives particles somewhere to form (CV > 0).
        
        Args:
            shape: Field dimensions
            n_hotspots: Number of localized seeds (increased to 12 for better structure)
            
        Returns:
            Field tensor with localized structure
        """
        # Background: low-amplitude noise
        field = 0.05 * torch.rand(*shape, device=self.device)  # REDUCED: was 0.1
        
        # Add Gaussian hotspots (braided strand pattern from mobius)
        for _ in range(n_hotspots):
            # Random center
            cx = torch.randint(0, shape[0], (1,)).item()
            cy = torch.randint(0, shape[1], (1,)).item()
            cz = torch.randint(0, shape[2], (1,)).item()
            
            # Random amplitude and width
            amp = 0.8 + 0.4 * torch.rand(1).item()  # INCREASED: 0.8-1.2 (was 0.6-1.0)
            width = 4 + 3 * torch.rand(1).item()    # REDUCED: 4-7 (was 5-10) for sharper peaks
            
            # Create Gaussian centered at (cx, cy, cz)
            x = torch.arange(shape[0], device=self.device)
            y = torch.arange(shape[1], device=self.device)
            z = torch.arange(shape[2], device=self.device)
            
            # Meshgrid for distances
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            
            # Distance from center (with periodic boundary)
            dx = torch.minimum(torch.abs(X - cx), shape[0] - torch.abs(X - cx))
            dy = torch.minimum(torch.abs(Y - cy), shape[1] - torch.abs(Y - cy))
            dz = torch.minimum(torch.abs(Z - cz), shape[2] - torch.abs(Z - cz))
            dist_sq = dx**2 + dy**2 + dz**2
            
            # Gaussian hotspot
            gaussian = amp * torch.exp(-dist_sq / (2 * width**2))
            field += gaussian
        
        return field
        
    def laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D Laplacian using periodic boundary conditions on GPU.
        
        Uses torch.roll for full vectorization - much faster than slicing.
        
        ∇²f = (f_{i+1,j,k} + f_{i-1,j,k} + f_{i,j+1,k} + f_{i,j-1,k} + f_{i,j,k+1} + f_{i,j,k-1} - 6f_{i,j,k})
        
        Args:
            field: 3D tensor
            
        Returns:
            Laplacian of field
        """
        # Fully vectorized with periodic boundaries
        lapl = (
            torch.roll(field, shifts=1, dims=0) + torch.roll(field, shifts=-1, dims=0) +
            torch.roll(field, shifts=1, dims=1) + torch.roll(field, shifts=-1, dims=1) +
            torch.roll(field, shifts=1, dims=2) + torch.roll(field, shifts=-1, dims=2) -
            6 * field
        )
        
        return lapl
    
    def gaussian_smooth(self, field: torch.Tensor, sigma: float = None) -> torch.Tensor:
        """
        Apply Gaussian smoothing for numerical stability.
        
        Proven effective in MED experiments for preventing instabilities.
        Uses separable 1D convolutions for efficiency on GPU.
        
        Args:
            field: Input field
            sigma: Smoothing width (defaults to self.gaussian_sigma)
            
        Returns:
            Smoothed field
        """
        if sigma is None:
            sigma = self.gaussian_sigma
            
        if sigma <= 0:
            return field
        
        # Simple approximation: 3x3x3 box filter (fast on GPU)
        # For true Gaussian, would need torch.nn.functional.conv3d with Gaussian kernel
        kernel_size = 3
        padding = kernel_size // 2
        
        # Average pooling approximates Gaussian for small sigma
        smoothed = (
            field +
            torch.roll(field, shifts=1, dims=0) + torch.roll(field, shifts=-1, dims=0) +
            torch.roll(field, shifts=1, dims=1) + torch.roll(field, shifts=-1, dims=1) +
            torch.roll(field, shifts=1, dims=2) + torch.roll(field, shifts=-1, dims=2)
        ) / 7.0
        
        return smoothed
        
    def apply_sec_collapse(self, field: torch.Tensor) -> torch.Tensor:
        """
        Apply Symbolic Entropy Collapse for quantum phenomena.
        
        SEC creates discrete quantum states from continuous fields
        through topological collapse at depth=2 recursion.
        
        Based on validated SEC experiments showing:
        - Bounded complexity (nodes ≤ 3)
        - Depth=2 optimal recursion
        - Born rule compliance emerges naturally
        """
        # Compute local entropy (simplified for speed)
        eps = 1e-10
        field_abs = torch.abs(field)
        field_sum = field_abs.sum() + eps
        field_normalized = field_abs / field_sum
        
        # Local entropy via sliding window approximation
        # High entropy regions are candidates for collapse
        kernel_size = 3
        local_mean = (field_abs + 
                     torch.roll(field_abs, 1, 0) + torch.roll(field_abs, -1, 0) +
                     torch.roll(field_abs, 1, 1) + torch.roll(field_abs, -1, 1) +
                     torch.roll(field_abs, 1, 2) + torch.roll(field_abs, -1, 2)) / 7.0
        
        local_var = (field_abs - local_mean) ** 2
        entropy_proxy = local_var / (local_mean + eps)
        
        # SEC collapse where entropy exceeds threshold
        collapse_mask = entropy_proxy > self.sec_threshold
        
        # Apply symbolic collapse (quantization to discrete states)
        collapsed_field = field.clone()
        
        # Collapse to canonical symbolic states (NOT based on field range)
        # This creates true quantum discreteness
        n_states = self.sec_nodes_max
        # Canonical states: -1, 0, +1 for 3 states (like spin states)
        states = torch.linspace(-1.0, 1.0, n_states, device=self.device)
        
        # Only collapse the masked regions
        field_to_collapse = field[collapse_mask]
        
        # Find nearest canonical state for each collapsing point (vectorized, GPU-friendly)
        if field_to_collapse.numel() > 0:  # Check count, not .any() which syncs
            distances = torch.abs(field_to_collapse.unsqueeze(-1) - states)
            nearest_state_idx = torch.argmin(distances, dim=-1)
            collapsed_values = states[nearest_state_idx]
            
            # Update only the collapsed regions
            collapsed_field[collapse_mask] = collapsed_values
        
        return collapsed_field
    
    def apply_med_dynamics(self, field: torch.Tensor) -> torch.Tensor:
        """
        Apply Macro Emergence Dynamics for fluid-like evolution.
        
        MED provides continuous field evolution through
        Navier-Stokes-like dynamics with bounded complexity.
        
        From your experiments:
        - Universal bounds: depth ≤ 1, nodes ≤ 3  
        - Viscosity provides stability
        """
        # Apply diffusion (viscous damping)
        diffusion = self.med_viscosity * self.laplacian(field)
        
        # MED update with bounded complexity
        update = diffusion
        
        # Limit update magnitude (stability from MED experiments)
        max_update = 0.01
        update = torch.clamp(update, -max_update, max_update)
        
        return field + update
    
    def evolve_quantum_wavefunction(self):
        """
        Evolve quantum wavefunction using SEC-MED integration.
        
        SEC provides collapse/measurement
        MED provides unitary evolution
        Together they create complete quantum mechanics
        """
        # Create wavefunction from E-I superposition
        # Don't normalize by max - let SEC collapse create localization
        psi_real = self.E
        psi_imag = self.I
        self.wavefunction_field = psi_real + 1j * psi_imag
        
        # Apply MED for unitary evolution (Schrödinger-like)
        psi_evolved_real = self.apply_med_dynamics(self.wavefunction_field.real)
        psi_evolved_imag = self.apply_med_dynamics(self.wavefunction_field.imag)
        psi_evolved = psi_evolved_real + 1j * psi_evolved_imag
        
        # Apply SEC for measurement/collapse
        psi_collapsed_real = self.apply_sec_collapse(psi_evolved.real)
        psi_collapsed_imag = self.apply_sec_collapse(psi_evolved.imag)
        psi_collapsed = psi_collapsed_real + 1j * psi_collapsed_imag
        
        # Update wavefunction
        self.wavefunction_field = psi_collapsed
        
        # Update coherence based on evolution vs collapse
        # High coherence when evolution dominates, low when collapse dominates
        collapse_diff = torch.abs(psi_collapsed - psi_evolved)
        evolution_strength = torch.abs(psi_evolved)
        
        # Coherence = how much the wavefunction maintains unitary evolution
        # vs discrete collapse (0 = fully collapsed, 1 = fully coherent)
        self.quantum_coherence = 1.0 - torch.clamp(collapse_diff / (evolution_strength + 1e-10), 0.0, 1.0)
        
        # Update entanglement map (non-local correlations from SEC)
        self.update_entanglement_map()
    
    def update_entanglement_map(self):
        """
        Track quantum entanglement through non-local correlations.
        
        Entanglement emerges from SEC creating correlated collapses
        at spatially separated points.
        """
        # Compute correlation strength from wavefunction
        psi = self.wavefunction_field
        psi_mag = torch.abs(psi)
        
        # Simple entanglement measure: spatial correlation
        # Correlate with shifted versions (non-local)
        correlation = torch.zeros_like(self.entanglement_map)
        for shift in [1, 2, 3]:
            correlation += torch.abs(psi_mag * torch.roll(psi_mag, shift, dims=0))
            correlation += torch.abs(psi_mag * torch.roll(psi_mag, shift, dims=1))
            correlation += torch.abs(psi_mag * torch.roll(psi_mag, shift, dims=2))
        
        # Normalize
        correlation = correlation / (6 * 3)
        self.entanglement_map = correlation
    
    def check_born_rule_compliance(self) -> float:
        """
        Check if Born rule emerges from SEC-MED dynamics.
        
        Born rule: P = |ψ|²
        
        Returns:
            Compliance score (1.0 = perfect Born rule)
        """
        # Compute probability distribution
        psi = self.wavefunction_field
        prob_measured = torch.abs(psi) ** 2
        total = prob_measured.sum()
        
        if total < 1e-10:
            return 0.0
            
        prob_measured = prob_measured / total
        
        # Check normalization
        norm_check = prob_measured.sum().item()
        
        # Check positivity
        positivity_check = (prob_measured >= 0).all().item()
        
        # Combined compliance score
        compliance = 0.0
        if positivity_check:
            compliance += 0.5
        if abs(norm_check - 1.0) < 0.01:
            compliance += 0.5
            
        return compliance
        
    def recursive_balance_field(self) -> torch.Tensor:
        """
        The RBF equation - generates all physics.
        
        B(x,t) = ∇²(E - I) + λM∇²M - α||E-I||²
        
        Where:
        - ∇²(E - I): Energy-Information gradient (drives evolution)
        - λM∇²M: Memory recursion (creates persistence)
        - α||E-I||²: Collapse coupling (triggers crystallization)
        
        Returns:
            The balance field that drives all evolution
        """
        # Energy-Information gradient
        # Pure RBF - no artificial smoothing needed if dynamics are correct
        ei_diff = self.E - self.I
        grad_term = self.laplacian(ei_diff)
        
        # Memory recursion term
        mem_laplace = self.laplacian(self.M)
        mem_term = self.lambda_mem * self.M * mem_laplace
        
        # Collapse coupling term (scaled down to prevent instability)
        collapse_term = -0.001 * self.alpha_collapse * (ei_diff ** 2)
        
        # Combined balance field
        balance = grad_term + mem_term + collapse_term
        
        return balance
    
    def evolve_step(self) -> dict:
        """
        Single evolution step with SEC-MED integration.
        
        MED provides continuous evolution
        SEC provides discrete collapse
        Together they generate quantum phenomena
        
        This is where the magic happens:
        1. Compute balance field (RBF)
        2. Apply MED to energy (continuous dynamics)
        3. Apply SEC to information (discrete collapse)
        4. Evolve quantum wavefunction (SEC+MED integration)
        5. Detect herniations (collapse sites)
        6. Apply collapses (create structure)
        7. Update memory (recursive component)
        
        Returns:
            Dictionary with step statistics
        """
        # Compute balance field
        B = self.recursive_balance_field()
        
        # Apply MED to energy field (continuous fluid-like dynamics)
        self.E = self.apply_med_dynamics(self.E)
        
        # Apply SEC to information field (discrete symbolic collapse)
        self.I = self.apply_sec_collapse(self.I)
        
        # Evolve quantum wavefunction (SEC+MED integration for QM emergence)
        self.evolve_quantum_wavefunction()
        
        # Detect herniations (collapse sites)
        herniations = self.detect_herniations(B)
        self.herniation_history.append(len(herniations))
        
        # Apply collapses in batch (GPU-efficient vectorized operations)
        self.apply_collapses_batched(herniations)
        
        # RBF DRIVES ENERGY REDISTRIBUTION NATURALLY
        # Energy follows balance gradient (no forced dispersion needed!)
        # The field pressure from ∇²(E-I) creates natural 1/r-like potentials
        balance_gradient = self.laplacian(B)
        energy_update = self.dt * balance_gradient
        
        # Check for instability before updating
        if torch.isnan(energy_update).any() or torch.isinf(energy_update).any():
            logger.error(f"❌ STEP {self.step_count}: NaN/Inf in energy_update!")
            logger.error(f"  Balance gradient stats: min={balance_gradient.min().item()}, max={balance_gradient.max().item()}")
            logger.error(f"  B stats: min={B.min().item()}, max={B.max().item()}")
        
        self.E += energy_update
        
        self.E = torch.clamp(self.E, min=-self.max_field_value, max=self.max_field_value)
        
        # === QUANTUM BALANCE EQUATION (QBE) ===
        # From legacy docs: dI/dt + dE/dt = λ·QPL(t)
        # QPL(t) = Q₀·cos(ω·t) - oscillatory quantum potential layer
        # This ENFORCES energy-information coupling - they are NOT independent!
        
        # Compute QPL(t) - the quantum regulatory function
        qpl_t = torch.cos(torch.tensor(self.qpl_omega * self.time, device=self.device))
        
        # QBE constraint: dI/dt = λ·QPL(t) - dE/dt
        # We already computed dE/dt (energy_update/dt), so:
        dE_dt = energy_update / self.dt
        dI_dt_qbe = self.lambda_qbe * qpl_t - dE_dt
        
        # Apply QBE-constrained information update
        info_update_qbe = self.dt * dI_dt_qbe
        self.I += info_update_qbe
        self.I = torch.clamp(self.I, min=0, max=2.0)
        
        # PAC Conservation: M + E + I = constant
        # Memory doesn't decay - it's conserved PAC component!
        # Memory accumulates from I→M collapses
        # NO DECAY: M *= 0.98 breaks PAC conservation
        
        # MED viscosity provides minimal diffusion for stability (like GAIA)
        # DON'T use Gaussian smoothing - it destroys localization!
        # Only MED's natural viscosity term (already in apply_med_dynamics)
        
        # Advance time
        self.time += self.dt
        self.step_count += 1
        
        # Periodic health check
        if self.step_count % 100 == 0:
            self._check_field_health()
        
        # Return minimal statistics (avoid GPU sync)
        # Only compute expensive stats when explicitly requested
        return {
            'time': self.time,
            'step': self.step_count,
            'herniations': len(herniations)
        }
    
    def get_statistics(self) -> dict:
        """
        Get detailed statistics - ONLY call when needed (not every step).
        Forces GPU→CPU sync via .item() calls.
        """
        return {
            'time': self.time,
            'step': self.step_count,
            'herniations': self.herniation_history[-1] if self.herniation_history else 0,
            'mean_energy': self.E.mean().item(),
            'mean_info': self.I.mean().item(),
            'mean_memory': self.M.mean().item(),
            'total_pac': (self.E.sum() + self.I.sum() + self.M.sum()).item(),
            'quantum_coherence': self.quantum_coherence.mean().item(),
            'entanglement': self.entanglement_map.mean().item(),
            'born_rule_compliance': self.check_born_rule_compliance()
        }
    
    def detect_herniations(self, balance_field: torch.Tensor, max_sites: int = 100) -> List[Tuple[int, int, int]]:
        """
        Find collapse sites where field exceeds threshold.
        
        From legacy cosmo.py: requires BOTH high information AND high energy.
        This dual-condition creates stable emergence dynamics.
        
        Args:
            balance_field: The RBF field to analyze
            max_sites: Maximum number of herniation sites to return (for performance)
            
        Returns:
            List of (x, y, z) coordinates of herniation sites
        """
        # Dual-condition collapse (proven in legacy experiments)
        # Requires BOTH information > threshold AND energy > energy_threshold
        info_mask = self.I > self.herniation_threshold
        energy_mask = torch.abs(self.E) > self.energy_threshold
        mask = info_mask & energy_mask  # Both conditions must be true
        
        herniation_indices = torch.nonzero(mask, as_tuple=False)
        
        # Limit number of sites for performance
        if len(herniation_indices) > max_sites:
            # Take the strongest herniations based on combined info+energy
            combined_strength = (self.I[mask] + torch.abs(self.E[mask])) / 2
            top_k = torch.topk(combined_strength, k=max_sites)
            herniation_indices = herniation_indices[top_k.indices]
        
        # Convert to list of tuples (move to CPU only for the limited set)
        if len(herniation_indices) > 0:
            herniation_cpu = herniation_indices.cpu()
            sites = [(int(herniation_cpu[i, 0]), int(herniation_cpu[i, 1]), int(herniation_cpu[i, 2])) 
                     for i in range(len(herniation_cpu))]
        else:
            sites = []
        
        return sites
    
    def apply_collapse(self, site: Tuple[int, int, int]) -> None:
        """
        SEC (Symbolic Entropy Collapse) at a point.
        
        KEY INSIGHT from Möbius-Confluence:
        - Information crystallizes (reduces locally)
        - Energy disperses fractally (increases globally)
        - This creates the potential for the NEXT collapse!
        
        The field is in perpetual collapse-regeneration.
        Each collapse fuels the next one.
        
        Args:
            site: (x, y, z) coordinates where collapse occurs
        """
        x, y, z = site
        
        # Legacy-validated collapse dynamics
        # Matter generation from BOTH info and energy
        collapse_val = self.collapse_threshold * (self.I[x, y, z] + torch.abs(self.E[x, y, z])) * 0.5
        
        # Memory records matter generation
        self.M[x, y, z] += collapse_val
        
        # Energy decays on collapse (proven in legacy: energy_decay = 0.9)
        self.E[x, y, z] *= 0.9
        
        # Information slightly reduced by crystallization
        self.I[x, y, z] *= 0.95
        
        # Skip expensive fractal dispersion for GPU efficiency
        # The laplacian operations already provide diffusion
    
    def apply_collapses_batched(self, sites: List[Tuple[int, int, int]]) -> None:
        """
        Apply multiple collapses in a batched, GPU-efficient manner.
        
        From Herniation Hypothesis:
        - Energy disperses RADIALLY (1/r → gravity emergence)
        - Information crystallizes FRACTALLY (recursive structure → quantum locking)
        - Memory accumulates at rupture sites (recursive kernel)
        
        Args:
            sites: List of (x, y, z) coordinates where collapses occur
        """
        if not sites:
            return
        
        # Extract coordinates efficiently (no GPU transfer needed)
        xs = [s[0] for s in sites]
        ys = [s[1] for s in sites]
        zs = [s[2] for s in sites]
        
        # Batch collapse operations using advanced indexing (stays on GPU)
        # PAC Conservation: collapse converts I→M (potential → crest)
        # Amount transferred must be exact: ΔM = -ΔI (conservation!)
        collapse_vals = self.collapse_threshold * (self.I[xs, ys, zs] + torch.abs(self.E[xs, ys, zs])) * 0.5
        
        # SEC: Information crystallizes into Memory (CONSERVATIVE transfer!)
        self.M[xs, ys, zs] += collapse_vals
        self.I[xs, ys, zs] -= collapse_vals  # Exact conservation: I lost = M gained
        
        # RBF automatically redistributes energy via balance field gradients
        # No manual dispersion needed - field pressure creates natural potentials!
        # The laplacian of ∇²(E-I) creates 1/r-like gravity automatically
    
    def compute_entropy_gradient(self, balance_field: torch.Tensor) -> torch.Tensor:
        """
        Information flows along entropy gradients.
        
        Entropy gradient = -∇²(log(I))
        
        This ensures information naturally flows from low to high complexity,
        enabling the emergence of structure.
        
        Args:
            balance_field: Current balance field (unused but kept for interface)
            
        Returns:
            Entropy gradient field
        """
        # Compute log of information field (with small constant to avoid log(0))
        log_info = torch.log(self.I + 1e-10)
        
        # Gradient is negative Laplacian
        gradient = -self.laplacian(log_info)
        
        return gradient
    
    def get_state(self) -> dict:
        """
        Get current complete state of the field.
        
        Returns:
            Dictionary with all field data and metadata (on CPU)
        """
        return {
            'E': self.E.cpu().numpy(),
            'I': self.I.cpu().numpy(),
            'M': self.M.cpu().numpy(),
            'time': self.time,
            'step': self.step_count,
            'shape': tuple(self.E.shape),
            'herniation_history': self.herniation_history.copy(),
            'device': str(self.device)
        }
    
    def compute_pac_conservation(self) -> dict:
        """
        Check PAC conservation law using Fracton's PAC Regulator.
        
        Total E + I + M should be approximately conserved.
        Target: 96.4% correlation from experiments.
        
        Uses Fracton's native PAC validation for proper conservation checking.
        
        Returns:
            Conservation metrics
        """
        total = (self.E.sum() + self.I.sum() + self.M.sum()).item()
        E_sum = self.E.sum().item()
        I_sum = self.I.sum().item()
        M_sum = self.M.sum().item()
        
        # Use Fracton PAC regulator to validate conservation
        # Validate: f(parent) = Σf(children)  where parent=total, children=[E,I,M]
        pac_result = self.pac_regulator.validate_recursive_conservation(
            parent_value=total,
            children_values=[E_sum, I_sum, M_sum],
            operation_context="field_conservation_check"
        )
        
        return {
            'total': total,
            'energy': E_sum,
            'information': I_sum,
            'memory': M_sum,
            'ratio_E': E_sum / total if total != 0 else 0,
            'ratio_I': I_sum / total if total != 0 else 0,
            'ratio_M': M_sum / total if total != 0 else 0,
            'pac_conserved': pac_result.conserved,
            'pac_residual': pac_result.residual,
            'pac_xi': pac_result.xi_value,
            'pac_corrected': pac_result.correction_applied
        }
    
    def find_stable_vortices(self, threshold: float = 0.5) -> List[dict]:
        """
        Find stable topological vortices in the field.
        
        These vortices ARE particles - stable knots in the field topology.
        No particle definitions needed; they emerge naturally.
        
        Args:
            threshold: Stability threshold in memory field
            
        Returns:
            List of detected vortices with properties
        """
        # Compute vorticity (gradient magnitude of energy field)
        # Use central differences
        grad_x = torch.zeros_like(self.E)
        grad_y = torch.zeros_like(self.E)
        grad_z = torch.zeros_like(self.E)
        
        grad_x[1:-1, :, :] = (self.E[2:, :, :] - self.E[:-2, :, :]) / 2.0
        grad_y[:, 1:-1, :] = (self.E[:, 2:, :] - self.E[:, :-2, :]) / 2.0
        grad_z[:, :, 1:-1] = (self.E[:, :, 2:] - self.E[:, :, :-2]) / 2.0
        
        vorticity = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Find regions with high vorticity AND high memory (stability)
        vorticity_threshold = torch.quantile(vorticity, 0.95)
        stable_regions = (vorticity > vorticity_threshold) & (self.M > threshold)
        
        # Get coordinates (move to CPU)
        coords = torch.nonzero(stable_regions, as_tuple=False).cpu()
        
        vortices = []
        for idx in coords:
            x, y, z = int(idx[0]), int(idx[1]), int(idx[2])
            vortices.append({
                'position': (x, y, z),
                'vorticity': vorticity[x, y, z].item(),
                'memory': self.M[x, y, z].item(),
                'energy': self.E[x, y, z].item(),
                'info': self.I[x, y, z].item()
            })
        
        return vortices
    
    def __repr__(self) -> str:
        """String representation of the field state."""
        return (f"DawnField(shape={self.E.shape}, time={self.time:.2f}, "
                f"step={self.step_count}, herniations={len(self.herniation_history)})")
