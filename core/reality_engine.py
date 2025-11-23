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
        dM_dt = self.rbf.alpha_collapse * (disequilibrium ** 2)
        
        # 5. Evolve fields using pure RBF+QBE dynamics
        E_new = E + self.dt * dE_dt_qbe
        I_new = I + self.dt * dI_dt_qbe
        M_new = M + self.dt * dM_dt
        
        # 6. CONFLUENCE: Geometric actualization via Möbius topology
        #    P_{t+1} = Ξ · A_t(u+π, 1-v)
        #    Blend to avoid shocking the system
        I_actualized = self.confluence.step(E_new, enforce_antiperiodicity=True)
        confluence_weight = 0.3  # 30% blend for stability
        I_new = (1.0 - confluence_weight) * I_new + confluence_weight * I_actualized
        
        # 7. Temperature EMERGES from disequilibrium
        T_new = torch.abs(E_new - I_new)
        
        # Only check for numerical errors (NaN/Inf)
        had_nan = torch.isnan(E_new).any() or torch.isnan(I_new).any() or torch.isnan(M_new).any()
        had_inf = torch.isinf(E_new).any() or torch.isinf(I_new).any() or torch.isinf(M_new).any()
        
        if had_nan or had_inf:
            print("WARNING: NaN/Inf detected - adjusting adaptive parameters")
            # Return previous state and let adaptive controller fix it
            return self._record_state()
        
        # 8. Create new state
        from substrate.field_types import FieldState
        
        # PAC conservation should EMERGE from RBF+QBE+Confluence geometry
        # NOT be enforced by rescaling. The Möbius topology and balance
        # dynamics naturally conserve PAC if equations are correct.
        
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


class AnalogExtension:
    """
    Analog Tensor Architecture Extension for Reality Engine.
    
    This wraps the existing tensor-based RealityEngine and adds hierarchical
    analog field centers that spawn dynamically from herniations.
    
    Key features:
    - Maintains existing engine as "macro" scale
    - Spawns finer-scale AnalogFieldCenters at intense herniations
    - PAC-couples scales via Xi operator
    - Computes fields on-demand using continuous equations
    
    Usage:
        engine = RealityEngine(size=(256, 128))
        engine.initialize('big_bang')
        
        # Wrap with analog extension
        analog = AnalogExtension(engine, enable_spawning=True)
        
        # Step as normal - analog centers spawn automatically
        analog.step()
    """
    
    def __init__(
        self,
        base_engine: RealityEngine,
        enable_spawning: bool = True,
        spawn_threshold: float = 0.5,
        max_centers: int = 10
    ):
        """
        Initialize analog extension around existing engine.
        
        Args:
            base_engine: Existing RealityEngine (becomes "macro" scale)
            enable_spawning: Whether to auto-spawn centers at herniations
            spawn_threshold: Herniation intensity threshold for spawning
            max_centers: Maximum number of analog centers
        """
        from core.analog_field_center import AnalogFieldCenter, create_quantum_center_at_herniation
        
        self.base_engine = base_engine
        self.enable_spawning = enable_spawning
        self.spawn_threshold = spawn_threshold
        self.max_centers = max_centers
        
        # Create macro-scale analog center wrapping base engine
        # Base engine operates at ~meter scale
        self.macro_center = AnalogFieldCenter(
            characteristic_scale=1.0,  # 1 meter
            bounds=(-100, 100, -100, 100),
            device=base_engine.device
        )
        
        # Child centers (finer scales) spawned from herniations
        self.child_centers: list = []
        
        # Statistics
        self.total_centers_spawned = 0
        self.coupling_events = 0
        
        print("🔬 Analog Architecture Extension activated")
        print(f"   Base engine: {base_engine.size}")
        print(f"   Spawning enabled: {enable_spawning}")
        print(f"   Spawn threshold: {spawn_threshold}")
    
    def step(self, herniation_data: Optional[Dict] = None) -> Dict:
        """
        Step the multi-scale system.
        
        1. Step base engine (macro scale)
        2. Check for herniation spawning
        3. Evolve child centers
        4. PAC-couple scales
        
        Args:
            herniation_data: Optional herniation info from HerniationDetector
            
        Returns:
            Combined state dictionary
        """
        # Step base engine
        base_state = self.base_engine.step()
        
        # Check for new center spawning
        if self.enable_spawning and herniation_data:
            self._check_spawn_centers(herniation_data)
        
        # Evolve child centers
        for center in self.child_centers:
            center.evolve(self.base_engine.dt)
        
        # Remove dead centers (energy depleted)
        self._remove_dead_centers()
        
        # PAC couple scales
        self._couple_scales()
        
        # Combine statistics
        state = base_state.copy()
        state['analog_stats'] = {
            'num_child_centers': len(self.child_centers),
            'total_spawned': self.total_centers_spawned,
            'coupling_events': self.coupling_events,
            'child_scales': [c.scale for c in self.child_centers]
        }
        
        return state
    
    def _check_spawn_centers(self, herniation_data: Dict):
        """
        Check if any herniations are intense enough to spawn new centers.
        Also dissipates herniation energy even if we can't spawn.
        
        Args:
            herniation_data: Dict with 'sites', 'intensity', 'count'
        """
        from core.analog_field_center import create_quantum_center_at_herniation
        
        if herniation_data['count'] == 0:
            return
        
        intensity = herniation_data.get('intensity', 0.0)
        
        # Intense enough to warrant action?
        if intensity > self.spawn_threshold:
            sites = herniation_data['sites']
            if len(sites) > 0:
                # Get first herniation site
                y, x = sites[0][0].item(), sites[0][1].item()
                
                # Herniations are natural - don't dissipate artificially
                # Let the RBF+QBE dynamics handle them
                
                # Try to spawn if under max
                if len(self.child_centers) < self.max_centers:
                    # Normalize to ±1 range
                    nu, nv = self.base_engine.size
                    norm_x = (x / nv) * 2 - 1
                    norm_y = (y / nu) * 2 - 1
                    
                    # Spawn quantum center
                    new_center = create_quantum_center_at_herniation(
                        herniation_position=(norm_x, norm_y),
                        herniation_intensity=intensity,
                        parent_center=self.macro_center,
                        device=self.base_engine.device
                    )
                    
                    self.child_centers.append(new_center)
                    self.total_centers_spawned += 1
                    
                    print(f"✨ Spawned analog center #{self.total_centers_spawned}")
                    print(f"   Position: ({norm_x:.3f}, {norm_y:.3f})")
                    print(f"   Scale: {new_center.scale:.2e} m")
                else:
                    print(f"⚠️  Max centers reached ({self.max_centers})")
    
    def _remove_dead_centers(self):
        """
        Remove centers that have decayed below viability threshold.
        This frees up slots for new centers to spawn.
        """
        initial_count = len(self.child_centers)
        
        # Debug: Show ages before filtering
        if initial_count > 0 and self.base_engine.step_count % 500 == 0:
            ages = [c.age for c in self.child_centers]
            energies = [c.total_energy for c in self.child_centers]
            print(f"📊 Center lifecycle check (step {self.base_engine.step_count}):")
            print(f"   Ages: {[f'{a:.1f}s' for a in ages]}")
            print(f"   Energies: {[f'{e:.3f}' for e in energies]}")
        
        # Filter out dead centers (energy < 1%)
        self.child_centers = [
            center for center in self.child_centers
            if center.total_energy > 0.01
        ]
        
        removed = initial_count - len(self.child_centers)
        if removed > 0:
            print(f"💀 Removed {removed} decayed analog center(s)")
            print(f"   Active centers: {len(self.child_centers)}/{self.max_centers}")
    
    def _couple_scales(self):
        """
        PAC-couple child centers to parent via Xi operator.
        
        This transfers conserved quantities between scales while
        maintaining PAC conservation.
        """
        xi_operator = 1.0571  # PAC balance constant
        
        for center in self.child_centers:
            center.couple_to_parent(xi_operator=xi_operator)
            self.coupling_events += 1
    
    def get_all_centers(self) -> list:
        """Get all field centers (macro + children)"""
        return [self.macro_center] + self.child_centers
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about analog system"""
        stats = {
            'macro_center': self.macro_center.get_statistics(),
            'child_centers': [c.get_statistics() for c in self.child_centers],
            'total_spawned': self.total_centers_spawned,
            'coupling_events': self.coupling_events,
            'total_memory_mb': sum(c._estimate_memory_usage() for c in self.get_all_centers())
        }
        return stats
    
    def cleanup_inactive_centers(self):
        """Remove centers that have become inactive"""
        active = []
        removed = 0
        
        for center in self.child_centers:
            if center.num_active_points > 10:  # Keep if still active
                active.append(center)
            else:
                removed += 1
        
        self.child_centers = active
        
        if removed > 0:
            print(f"🗑️  Removed {removed} inactive analog centers")


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
