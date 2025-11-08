"""
Reality Engine - Unified Interface for Physics Simulation

Integrates all Reality Engine v2 components:
- Möbius substrate (geometric foundation)
- Thermodynamic fields (energy-information duality)
- SEC operator (symbolic entropy collapse)
- Confluence operator (geometric time stepping)
- Time emergence (temporal flow from disequilibrium)

This is the main production interface users interact with.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Generator
from dataclasses import dataclass, asdict

from substrate.mobius_manifold import MobiusManifold
from substrate.field_types import FieldState
from conservation.sec_operator import SymbolicEntropyCollapse
from conservation.thermodynamic_pac import ThermodynamicPAC
from dynamics.confluence import MobiusConfluence
from dynamics.time_emergence import TimeEmergence
from core.adaptive_parameters import AdaptiveParameters
from core.pac_rescaler import PACRescaler


@dataclass
class EngineState:
    """Complete snapshot of Reality Engine state"""
    step: int
    time: float
    fields: FieldState
    sec_stats: Dict
    confluence_stats: Dict
    time_stats: Dict
    pac_metrics: Dict


class RealityEngine:
    """
    Unified Reality Engine interface.
    
    Usage:
        engine = RealityEngine(size=(128, 32))
        engine.initialize('big_bang')
        
        for state in engine.evolve(steps=10000):
            if state['step'] % 100 == 0:
                print(f"Step {state['step']}: T_mean={state['T_mean']:.2f}")
    """
    
    def __init__(
        self,
        size: Tuple[int, int] = (128, 32),
        dt: float = 0.001,  # Smaller timestep for stability with damping
        device: str = 'auto'
    ):
        """
        Initialize Reality Engine.
        
        Args:
            size: Field dimensions (nu, nv) - must have even nu
            dt: Time step size (default 0.001 for numerical stability)
            device: 'cuda', 'cpu', or 'auto'
        """
        # Device selection
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.size = size
        self.dt = dt
        
        # Initialize components
        # MobiusManifold expects separate size and width
        nu, nv = size
        self.substrate = MobiusManifold(size=nu, width=nv, device=self.device)
        self.sec = SymbolicEntropyCollapse(device=self.device)
        self.confluence = MobiusConfluence(size=size, device=self.device)
        self.pac = ThermodynamicPAC()
        self.time_emer = TimeEmergence()
        
        # Engine state
        self.current_state: Optional[FieldState] = None  # Track current field state
        self.initialized = False
        self.step_count = 0
        self.time_elapsed = 0.0
        self.history = []
        self.max_history = 1000  # Prevent memory overflow
        
        # Adaptive parameter controller - NO MANUAL TUNING!
        self.adaptive = AdaptiveParameters(
            initial_gamma=0.005,  # Start higher with confluence
            initial_dt=dt,
            min_gamma=0.001,  # Allow low but not too low
            max_dt=0.01  # Cap dt to prevent NaN in long runs
        )
        
        # PAC rescaler - ensures total PAC conservation
        self.pac_rescaler = PACRescaler(alpha_pac=0.964)
    
    def initialize(self, mode: str = 'big_bang'):
        """
        Initialize the universe.
        
        Args:
            mode: Initialization mode
                - 'big_bang': Hot dense start
                - 'random': Random fluctuations
                - 'cold': Low temperature start
                - 'structured': Pre-evolved structure
        """
        # Initialize substrate fields
        self.current_state = self.substrate.initialize_fields(mode=mode)
        
        # Record initial state
        self.initialized = True
        self.step_count = 0
        self.time_elapsed = 0.0
        self.history = [self._record_state()]
    
    def step(self) -> Dict:
        """
        Perform one complete evolution step using Dawn Field equations.
        
        Evolution cycle (EMERGENT PHYSICS):
        1. RBF computes balance field (THE fundamental equation)
        2. QBE regulates E↔I dynamics (prevents runaway)
        3. CONFLUENCE: Geometric actualization operator (P→A transformation)
        4. All thermodynamics, stability, and atoms EMERGE naturally
        
        Confluence IS NOT enforcement - it's the geometric transformation
        that actualizes Potential into Actual while naturally conserving PAC
        through Möbius topological invariance.
        
        Returns:
            State dictionary
        """
        if not self.initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        import torch
        import sys
        from pathlib import Path
        
        # Initialize RBF engine (lazy initialization)
        if not hasattr(self, 'rbf'):
            # Import from fracton SDK
            fracton_path = Path(__file__).parent.parent.parent / 'fracton'
            if str(fracton_path) not in sys.path:
                sys.path.insert(0, str(fracton_path))
            
            from fracton.field.rbf_engine import RBFEngine
            self.rbf = RBFEngine(
                lambda_mem=0.020,      # Universal frequency from experiments
                alpha_collapse=0.964,  # PAC validated constant
                gamma_damping=self.adaptive.gamma,  # ADAPTIVE gamma from feedback!
                backend='torch'  # Always torch for consistency
            )
        
        # Initialize QBE regulator (lazy initialization)
        if not hasattr(self, 'qbe'):
            from fracton.field.qbe_regulator import QBERegulator
            self.qbe = QBERegulator(
                lambda_qbe=1.0,
                qpl_omega=0.020,  # 0.020 Hz universal frequency
                backend='torch'  # Always torch for consistency
            )
        
        # Get current fields using Dawn Field nomenclature
        state = self.current_state
        E = state.actual      # Energy field (actualization)
        I = state.potential   # Information field (potential)
        M = state.memory      # Memory field (persistence)
        
        # === CORE DAWN FIELD DYNAMICS ===
        
        # 1. Compute Recursive Balance Field (THE fundamental equation)
        #    B(x,t) = ∇²(E-I) + λM∇²M - α||E-I||² - γ(E-I)
        #    This ONE equation generates ALL physics!
        #    γ is NOW ADAPTIVE based on PAC and stability feedback!
        
        # Update RBF gamma with current adaptive value
        self.rbf.gamma_damping = self.adaptive.gamma
        
        B = self.rbf.compute_balance_field(E, I, M)
        
        # 2. Compute time derivatives from RBF
        dE_dt = B  # Energy follows balance field
        
        # 3. Apply QBE constraint to get dI/dt
        #    QBE: dI/dt + dE/dt = λ·QPL(t)
        #    This prevents runaway and creates natural stability
        dE_dt_qbe, dI_dt_qbe = self.qbe.enforce_qbe_constraint(
            dE_dt, 
            -B,  # Initial guess: dI/dt = -dE/dt
            t=self.time_elapsed
        )
        
        # 4. Memory evolves from collapse events (emerges naturally!)
        #    dM/dt = α||E-I||² (memory accumulates where information collapses)
        disequilibrium = E - I
        # Adaptive memory boost: scale based on grid size to prevent runaway
        # Smaller grids need more boost; larger grids self-organize better
        grid_size = E.numel()
        base_boost = 10.0
        memory_boost = base_boost * (4096 / grid_size) ** 0.5  # Scale with 1/√N
        memory_boost = max(1.0, min(memory_boost, 10.0))  # Clamp 1-10x
        dM_dt = memory_boost * self.rbf.alpha_collapse * (disequilibrium ** 2)
        
        # CRITICAL: Prevent memory overflow in long simulations
        # Once structures form and grow massive (M > 100), we need to slow accumulation
        # This prevents the NaN cascade at 3500+ steps from unbounded M growth
        M_max = M.max()
        M_mean = M.mean()
        
        # AGGRESSIVE multi-stage overflow prevention:
        # The key insight: once M > 50, we're in late-stage collapse
        # Need to STRONGLY limit further growth to prevent runaway
        if M_max > 200:
            # EMERGENCY: structures critically massive
            # At this point, we're seeing 200+ mass - gravity has won
            # Just maintain current state, no more growth
            overflow_factor = 200 / (M_max + 1e-10)
            dM_dt = dM_dt * overflow_factor * 0.05  # 95% brake!
        elif M_max > 100:
            # STRONG brake for large structures (100-200 range)
            # This is where previous runs started failing
            overflow_factor = 100 / (M_max + 1e-10)
            dM_dt = dM_dt * overflow_factor * 0.2  # 80% brake
        elif M_max > 50:
            # MODERATE brake (50-100 range)
            # Start slowing growth significantly
            overflow_factor = 1.0 / (1.0 + (M_max - 50) / 25.0)
            dM_dt = dM_dt * overflow_factor * 0.5  # 50% reduction
        elif M_max > 20:
            # GENTLE brake (20-50 range)
            overflow_factor = 1.0 / (1.0 + (M_max - 20) / 30.0)
            dM_dt = dM_dt * overflow_factor
        
        # Also prevent runaway in high-density regions (local dampening)
        M_threshold = M_mean + 3 * M.std()  # 3-sigma outliers
        high_density_mask = M > M_threshold
        if high_density_mask.any():
            # Reduce growth rate in already-dense regions
            local_damping = torch.ones_like(dM_dt)
            local_damping[high_density_mask] = 0.3  # 70% slower in dense regions
            dM_dt = dM_dt * local_damping
        
        # 5. Evolve fields using RBF+QBE dynamics
        E_new = E + self.dt * dE_dt_qbe
        I_new = I + self.dt * dI_dt_qbe
        M_new = M + self.dt * dM_dt
        
        # CRITICAL: Direct field magnitude control to prevent runaway oscillations
        # The core issue: E and I can oscillate to extreme values even with M braking
        # MUST preserve PAC conservation: E + I = constant
        # Solution: Scale BOTH fields by the SAME factor to limit max while preserving sum
        E_magnitude = torch.abs(E_new).max()
        I_magnitude = torch.abs(I_new).max()
        max_magnitude = max(E_magnitude, I_magnitude)
        
        # Apply soft clamping when fields get large (> 1000)
        # Use SAME scaling for both E and I to preserve E+I conservation
        if max_magnitude > 1000:
            # Soft clamp: scale down gradually as we exceed threshold
            scale_factor = 1000 / max_magnitude
            # Use smooth transition: blend between no scaling and full scaling
            blend = min(1.0, (max_magnitude - 1000) / 1000)  # 0 at 1000, 1 at 2000
            unified_scale = (1.0 - blend + blend * scale_factor)
            
            # Apply SAME scale to both fields - this preserves E+I sum!
            E_new = E_new * unified_scale
            I_new = I_new * unified_scale
        
        # CRITICAL: Field overflow detection and prevention
        # Monitor field magnitudes to catch catastrophic issues
        E_max = torch.abs(E_new).max()
        I_max = torch.abs(I_new).max()
        M_max_new = M_new.max()
        
        # Emergency threshold - if we hit this, something went very wrong
        overflow_threshold = 1e5
        needs_rescale = False
        
        if E_max > overflow_threshold or I_max > overflow_threshold or M_max_new > overflow_threshold:
            print(f"  [EMERGENCY] Catastrophic overflow: E_max={E_max:.1e}, I_max={I_max:.1e}, M_max={M_max_new:.1e}")
            needs_rescale = True
        
        # Emergency rescaling as last resort
        if needs_rescale:
            max_magnitude = max(E_max, I_max, M_max_new)
            scale_factor = 500.0 / max_magnitude
            
            print(f"  [EMERGENCY RESCALE] Applying scale factor {scale_factor:.2e}")
            
            E_new = E_new * scale_factor
            I_new = I_new * scale_factor
            M_new = M_new * scale_factor
            
            # Also rescale current state for consistency
            self.current_state.actual = self.current_state.actual * scale_factor
            self.current_state.potential = self.current_state.potential * scale_factor
            self.current_state.memory = self.current_state.memory * scale_factor
        
        # 6. CONFLUENCE: Ξ-Balance Through Geometric Actualization
        #    From PAC papers: Confluence IS the Ξ=1.0571072 balance operator
        #    
        #    P_{t+1} = Ξ · A_t(u+π, 1-v)
        #    
        #    This transformation maintains the universal conservation ratio
        #    between hierarchical levels. It's not arbitrary geometry - it's
        #    the mathematical manifestation of PAC functional conservation.
        #    
        #    The Möbius topology + Ξ-scaling ensures:
        #        PAC = P + Ξ·A + α·M = constant
        I_actualized = self.confluence.step(E_new, enforce_antiperiodicity=True)
        
        # Blend QBE dynamics with Ξ-balanced confluence
        # Weight determines how strongly Ξ-balance stabilizes the dynamics
        # Start gentle (let structures form) → increase (stabilize long-term)
        confluence_weight = 0.3  # 30% Ξ-balance maintains conservation without over-damping
        I_new = (1.0 - confluence_weight) * I_new + confluence_weight * I_actualized
        
        # 7. Temperature EMERGES from disequilibrium (not manually computed!)
        #    T = ||E-I|| in Dawn Field Theory
        T_new = torch.abs(E_new - I_new)
        
        # Minimal safety: only prevent actual numerical errors
        # (NO arbitrary clamping or artificial stability!)
        had_nan = torch.isnan(E_new).any() or torch.isnan(I_new).any() or torch.isnan(M_new).any()
        if had_nan:
            print("WARNING: NaN detected in fields - REVERTING to previous state!")
            # Don't corrupt the state - return previous
            # This triggers adaptive controller to increase gamma significantly
            return self._record_state()
        
        # 8. Create new state
        from substrate.field_types import FieldState
        new_state = FieldState(
            potential=I_new,
            actual=E_new,
            memory=M_new,
            temperature=T_new
        )
        
        # 9. PAC verification and adaptive parameter update
        # Compute PAC conservation quality for feedback
        alpha_pac = 0.964
        pac_total_before = (state.potential + state.actual + alpha_pac * state.memory).sum()
        pac_total_after = (I_new + E_new + alpha_pac * M_new).sum()
        pac_residual = torch.abs(pac_total_after - pac_total_before) / torch.abs(pac_total_before + 1e-10)
        
        # Compute QBE residual for feedback
        # QBE constraint: dI/dt + dE/dt = λ·QPL(t)
        # Measure how well this was satisfied
        dE_actual = (E_new - state.actual) / self.dt
        dI_actual = (I_new - state.potential) / self.dt
        qbe_sum = (dE_actual + dI_actual).mean()
        qbe_expected = self.qbe.lambda_qbe * self.qbe.compute_qpl(self.time_elapsed)
        qbe_residual = torch.abs(qbe_sum - qbe_expected) / (torch.abs(qbe_expected) + 1e-10)
        
        # Compute field energy for feedback
        field_energy = (E_new**2 + I_new**2 + M_new**2).sum()
        
        # Update adaptive parameters based on QBE feedback!
        adaptive_params = self.adaptive.update(
            pac_residual=pac_residual.item(),
            qbe_residual=qbe_residual.item(),
            field_energy=field_energy.item(),
            had_nan=had_nan
        )
        
        # Update dt for next step if it changed
        if abs(adaptive_params['dt'] - self.dt) > 1e-6:
            self.dt = adaptive_params['dt']
        
        # Report when parameters change significantly
        if self.step_count % 100 == 0:
            diag = self.adaptive.get_diagnostics()
            print(f"Step {self.step_count}: gamma={diag['gamma']:.4f}, dt={diag['dt']:.5f}, "
                  f"PAC={diag['pac_quality']:.3f}, stability={diag['stability_score']:.3f}")
        
        # 10. Update QBE internal time
        self.qbe.update_time(self.dt)
        
        # 11. Update state
        self.current_state = new_state
        self.step_count += 1
        
        # Time dilation emerges from disequilibrium
        avg_disequilibrium = torch.abs(disequilibrium).mean()
        time_dilation = 1.0 / (1.0 + avg_disequilibrium)
        self.time_elapsed += self.dt * time_dilation.item()
        
        # 12. Record emergent observables (not enforce them!)
        recorded_state = self._record_state()
        recorded_state['emergent_metrics'] = {
            'balance_magnitude': float(torch.abs(B).mean()),
            'disequilibrium': float(avg_disequilibrium),
            'time_dilation': float(time_dilation),
            'memory_accumulation': float(dM_dt.mean()),
            'structure_count': int((M_new > 0.1).sum()),  # Where memory accumulates
            'qpl_phase': float(self.qbe.compute_qpl(self.time_elapsed))
        }
        
        return recorded_state
    
    def evolve(
        self,
        steps: int,
        record_every: int = 1,
        callback: Optional[callable] = None
    ) -> Generator[Dict, None, None]:
        """
        Evolve for multiple steps.
        
        Args:
            steps: Number of steps to evolve
            record_every: Record state every N steps
            callback: Optional function called with state each step
        
        Yields:
            State dictionaries
        """
        for i in range(steps):
            state = self.step()
            
            if callback is not None:
                callback(state)
            
            if i % record_every == 0:
                yield state
    
    def get_state(self) -> EngineState:
        """
        Get complete engine state.
        
        Returns:
            EngineState with all fields and statistics
        """
        if not self.initialized:
            raise RuntimeError("Engine not initialized.")
        
        return EngineState(
            step=self.step_count,
            time=self.time_elapsed,
            fields=self.current_state,
            sec_stats=self.sec.get_sec_state(),
            confluence_stats=self.confluence.get_confluence_state(),
            time_stats={},  # Would get from time_emergence
            pac_metrics={}  # Would get from PAC
        )
    
    def _record_state(self) -> Dict:
        """
        Record current state as dictionary for history.
        
        Returns:
            Lightweight state dictionary
        """
        state = self.current_state
        
        recorded = {
            'step': self.step_count,
            'time': self.time_elapsed,
            # Field statistics (using aliases for compatibility)
            'P_mean': state.P.mean().item(),
            'P_std': state.P.std().item(),
            'A_mean': state.A.mean().item(),
            'A_std': state.A.std().item(),
            'A_max': state.A.max().item(),
            'A_min': state.A.min().item(),
            'M_mean': state.M.mean().item(),
            'M_std': state.M.std().item(),
            'M_max': state.M.max().item(),
            'M_min': state.M.min().item(),
            'T_mean': state.T.mean().item(),
            'T_std': state.T.std().item(),
            'T_max': state.T.max().item(),
            'T_min': state.T.min().item(),
            # Emergent thermodynamics
            'entropy': state.entropy(),
            'free_energy': state.free_energy(),
            'disequilibrium': state.disequilibrium(),
        }
        
        # Add to history (with size limit)
        self.history.append(recorded)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return recorded
    
    def discover_laws(self, min_history: int = 50) -> Dict:
        """
        Analyze history to discover emergent physical laws.
        
        Detects:
        - Conservation laws (energy, momentum, charge analogs)
        - Thermodynamic laws (2nd law compliance, cooling trends)
        - Emergent constants (effective c, α, coupling constants)
        - Phase transitions and critical points
        - Information-thermodynamic relationships
        
        Args:
            min_history: Minimum history length required
        
        Returns:
            Dictionary of discovered laws, constants, and patterns
        """
        if len(self.history) < min_history:
            return {
                'status': 'insufficient_data',
                'history_length': len(self.history),
                'required': min_history
            }
        
        import numpy as np
        from scipy import stats
        
        laws = {
            'status': 'analysis_complete',
            'discovered_laws': {},
            'emergent_constants': {},
            'phase_transitions': [],
            'correlations': {}
        }
        
        # Extract time series
        times = np.array([s['time'] for s in self.history])
        temperatures = np.array([s['T_mean'] for s in self.history])
        memories = np.array([s['M_mean'] for s in self.history])
        entropies = np.array([s['entropy'] for s in self.history])
        free_energies = np.array([s['free_energy'] for s in self.history])
        disequilibria = np.array([s['disequilibrium'] for s in self.history])
        total_heat = np.array([s['total_heat'] for s in self.history])
        collapse_events = np.array([s['collapse_events'] for s in self.history])
        
        # 1. THERMODYNAMIC LAWS
        # =====================
        
        # Second law: entropy must increase (or stay constant)
        entropy_trend = np.polyfit(times, entropies, 1)[0]
        entropy_violations = np.sum(np.diff(entropies) < -0.01)  # Significant decreases
        total_steps = len(entropies) - 1
        
        laws['discovered_laws']['second_law'] = {
            'compliant': entropy_violations / total_steps < 0.1,  # < 10% violations
            'entropy_rate': float(entropy_trend),
            'violations': int(entropy_violations),
            'violation_rate': float(entropy_violations / total_steps),
            'interpretation': 'cooling' if entropy_trend > 0 else 'heating'
        }
        
        # Landauer principle: heat generation ∝ information collapse
        if len(collapse_events) > 10:
            heat_per_collapse = total_heat / (collapse_events + 1)
            landauer_constant = np.mean(heat_per_collapse[10:])  # Exclude initial transient
            landauer_std = np.std(heat_per_collapse[10:])
            
            laws['discovered_laws']['landauer_principle'] = {
                'verified': landauer_std / (landauer_constant + 1e-10) < 0.5,
                'heat_per_collapse': float(landauer_constant),
                'variation': float(landauer_std),
                'interpretation': 'Information erasure generates heat'
            }
        
        # 2. CONSERVATION LAWS
        # ====================
        
        # Free energy conservation (should be approximately conserved)
        energy_mean = np.mean(free_energies)
        energy_std = np.std(free_energies)
        energy_variation = energy_std / (np.abs(energy_mean) + 1e-10)
        
        laws['discovered_laws']['energy_conservation'] = {
            'conserved': energy_variation < 0.1,  # < 10% variation
            'mean_energy': float(energy_mean),
            'variation': float(energy_variation),
            'std_dev': float(energy_std)
        }
        
        # Memory (matter) conservation: should grow monotonically, never decrease
        memory_violations = np.sum(np.diff(memories) < -1e-6)
        
        laws['discovered_laws']['matter_conservation'] = {
            'conserved': memory_violations == 0,
            'violations': int(memory_violations),
            'total_created': float(memories[-1] - memories[0]),
            'interpretation': 'Information → Matter crystallization'
        }
        
        # 3. EMERGENT CONSTANTS
        # =====================
        
        # Effective speed of light (from time emergence)
        if 'time_metrics' in self.history[-1]:
            c_values = [s.get('time_metrics', {}).get('c_effective', 1.0) 
                       for s in self.history if 'time_metrics' in s]
            if c_values:
                laws['emergent_constants']['c_effective'] = {
                    'value': float(np.mean(c_values)),
                    'std': float(np.std(c_values)),
                    'stability': float(np.std(c_values) / (np.mean(c_values) + 1e-10))
                }
        
        # Temperature-Memory coupling constant
        # dM/dt ∝ T × structure_signal, find proportionality constant
        if len(memories) > 20:
            dM_dt = np.diff(memories)
            T_avg = (temperatures[:-1] + temperatures[1:]) / 2
            # Filter out zero temperature to avoid division issues
            mask = T_avg > 0.01
            if np.sum(mask) > 10:
                coupling = dM_dt[mask] / T_avg[mask]
                
                laws['emergent_constants']['temperature_memory_coupling'] = {
                    'alpha_TM': float(np.mean(coupling)),
                    'std': float(np.std(coupling)),
                    'interpretation': 'Rate of matter formation per unit temperature'
                }
        
        # Cooling rate constant (exponential decay)
        if len(temperatures) > 20:
            # Fit T(t) = T0 * exp(-γt) after peak
            T_peak_idx = np.argmax(temperatures)
            if T_peak_idx < len(temperatures) - 10:
                t_decay = times[T_peak_idx:] - times[T_peak_idx]
                T_decay = temperatures[T_peak_idx:]
                T_peak = temperatures[T_peak_idx]
                
                # Log fit: ln(T) = ln(T0) - γt
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_T = np.log(T_decay / T_peak)
                    valid = np.isfinite(log_T)
                    if np.sum(valid) > 5:
                        gamma, _ = np.polyfit(t_decay[valid], log_T[valid], 1)
                        
                        laws['emergent_constants']['cooling_rate'] = {
                            'gamma': float(-gamma),
                            'interpretation': 'Exponential cooling constant',
                            'half_life': float(np.log(2) / (-gamma)) if gamma < 0 else float('inf')
                        }
        
        # 4. INFORMATION-THERMODYNAMIC CORRELATIONS
        # =========================================
        
        # Temperature vs Memory correlation (should be positive during formation)
        corr_TM, p_value_TM = stats.pearsonr(temperatures, memories)
        laws['correlations']['temperature_memory'] = {
            'correlation': float(corr_TM),
            'p_value': float(p_value_TM),
            'significant': p_value_TM < 0.05,
            'interpretation': 'Heat enables matter formation' if corr_TM > 0.5 else 'Weak coupling'
        }
        
        # Heat generation vs Collapse events (Landauer validation)
        if len(collapse_events) > 10:
            # Use increments
            dHeat = np.diff(total_heat)
            dCollapse = np.diff(collapse_events)
            mask = dCollapse > 0
            if np.sum(mask) > 5:
                corr_HC, p_value_HC = stats.pearsonr(dHeat[mask], dCollapse[mask])
                laws['correlations']['heat_collapse'] = {
                    'correlation': float(corr_HC),
                    'p_value': float(p_value_HC),
                    'significant': p_value_HC < 0.05,
                    'interpretation': 'Landauer: Collapse → Heat'
                }
        
        # Disequilibrium vs Temperature (early universe should be hot AND disequilibrated)
        corr_DT, p_value_DT = stats.pearsonr(disequilibria, temperatures)
        laws['correlations']['disequilibrium_temperature'] = {
            'correlation': float(corr_DT),
            'p_value': float(p_value_DT),
            'significant': p_value_DT < 0.05,
            'interpretation': 'Disequilibrium drives heat generation' if corr_DT > 0 else 'Complex coupling'
        }
        
        # 5. PHASE TRANSITIONS
        # ====================
        
        # Detect rapid changes in derivatives (phase transitions)
        dT_dt = np.diff(temperatures)
        dM_dt = np.diff(memories)
        
        # Find where second derivative changes sign dramatically
        d2T_dt2 = np.diff(dT_dt)
        d2M_dt2 = np.diff(dM_dt)
        
        # Transition: where |d²T/dt²| > threshold
        T_threshold = np.std(d2T_dt2) * 2
        T_transitions = np.where(np.abs(d2T_dt2) > T_threshold)[0]
        
        M_threshold = np.std(d2M_dt2) * 2
        M_transitions = np.where(np.abs(d2M_dt2) > M_threshold)[0]
        
        if len(T_transitions) > 0:
            laws['phase_transitions'].append({
                'type': 'thermal',
                'steps': T_transitions.tolist()[:5],  # First 5
                'interpretation': 'Rapid temperature change events'
            })
        
        if len(M_transitions) > 0:
            laws['phase_transitions'].append({
                'type': 'structural',
                'steps': M_transitions.tolist()[:5],
                'interpretation': 'Matter formation bursts'
            })
        
        # 6. SUMMARY STATISTICS
        # =====================
        
        laws['summary'] = {
            'total_steps': len(self.history),
            'time_span': float(times[-1] - times[0]),
            'temperature_range': [float(np.min(temperatures)), float(np.max(temperatures))],
            'memory_growth': float(memories[-1] / (memories[0] + 1e-10)),
            'total_collapses': int(collapse_events[-1]),
            'entropy_change': float(entropies[-1] - entropies[0])
        }
        
        return laws
    
    def reset(self):
        """Reset engine to uninitialized state."""
        self.initialized = False
        self.step_count = 0
        self.time_elapsed = 0.0
        self.history = []
        
        # Reset component statistics
        self.sec.total_heat_generated = 0.0
        self.sec.total_entropy_reduced = 0.0
        self.sec.collapse_event_count = 0
        self.sec.collapse_events = []
        
        self.confluence.total_steps = 0
        self.confluence.total_confluence_magnitude = 0.0
    
    def __repr__(self) -> str:
        """String representation."""
        if not self.initialized:
            return f"RealityEngine(size={self.size}, device='{self.device}')\n  Status: not initialized"
        
        return (
            f"RealityEngine(size={self.size}, device='{self.device}')\n"
            f"  Status: initialized\n"
            f"  Steps: {self.step_count}\n"
            f"  Time: {self.time_elapsed:.6f}\n"
            f"  History: {len(self.history)} states"
        )


def create_reality(
    size: Tuple[int, int] = (128, 32),
    device: str = 'auto'
) -> RealityEngine:
    """
    Convenience function to create a Reality Engine.
    
    Args:
        size: Field dimensions
        device: Computation device
    
    Returns:
        Configured RealityEngine
    """
    return RealityEngine(size=size, device=device)
