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
        dt: float = 0.001,
        device: str = 'auto'
    ):
        """
        Initialize Reality Engine.
        
        Args:
            size: Field dimensions (nu, nv) - must have even nu
            dt: Time step size
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
        Perform one complete evolution step.
        
        Evolution cycle:
        1. SEC evolves A toward P (thermodynamic coupling)
        2. Generate heat from collapse
        3. Confluence creates new P from A (geometric time step)
        4. PAC enforces conservation
        5. Time emerges from disequilibrium
        
        Returns:
            State dictionary
        """
        if not self.initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        # Get current state
        state = self.current_state
        
        # 1. SEC Evolution: A evolves toward P with thermodynamic coupling
        A_new, heat_generated = self.sec.evolve(
            state.A,
            state.P,
            state.T,
            dt=self.dt,
            add_thermal_noise=True
        )
        
        # 2. Update temperature with heat generation, diffusion, and cooling
        import torch
        import torch.nn.functional as F
        
        # Add heat uniformly (Landauer principle - collapse generates heat)
        heat_per_cell = heat_generated / state.T.numel()
        T_with_heat = state.T + heat_per_cell
        
        # Apply thermal diffusion (Fourier's law: heat flows from hot to cold)
        thermal_alpha = 0.5  # Strong diffusion coefficient
        
        # Compute Laplacian for diffusion
        # PyTorch pad needs 4D: [batch, channel, height, width]
        T_4d = T_with_heat.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        T_padded_4d = F.pad(T_4d, (1, 1, 1, 1), mode='circular')
        T_padded = T_padded_4d.squeeze(0).squeeze(0)  # Back to [H, W]
        
        laplacian = (
            T_padded[2:, 1:-1] +   # up
            T_padded[:-2, 1:-1] +  # down
            T_padded[1:-1, 2:] +   # right
            T_padded[1:-1, :-2] -  # left
            4 * T_with_heat        # center
        )
        
        # Apply diffusion step
        T_diffused = T_with_heat + thermal_alpha * self.dt * laplacian
        
        # Apply cooling that scales with temperature
        # Balance cooling with heat generation: at equilibrium dT/dt = 0
        # Heat added per cell per step: ~11.6/256 ≈ 0.045
        # Cooling per cell: γ * T * dt = 0.7 * T * 0.1 = 0.07 * T
        # Equilibrium when: 0.045 = 0.07 * T_eq => T_eq ≈ 0.64
        # To get T_eq ~ 5-10, reduce cooling coefficient
        cooling_coefficient = 0.85  # Fine-tuned for equilibrium
        T_new = T_diffused * (1.0 - cooling_coefficient * self.dt)
        
        # Prevent negative temperatures (physical constraint)
        T_new = torch.clamp(T_new, min=0.01)
        
        # 3. Confluence: Create new P from current A (geometric time step)
        P_new = self.confluence.step(A_new, enforce_antiperiodicity=True)
        
        # 4. Memory accumulation: Information → Matter crystallization
        # Memory grows where A has collapsed (low entropy, high structure)
        # Compute local "structuredness" = -entropy = pattern strength
        local_variance = (A_new - A_new.mean()).pow(2)
        structure_signal = torch.exp(-local_variance)  # High where A is stable
        
        # Memory accumulates from structure, decays slowly
        memory_growth_rate = 0.01
        memory_decay_rate = 0.001
        M_new = state.M + memory_growth_rate * structure_signal * self.dt
        M_new = M_new * (1.0 - memory_decay_rate * self.dt)  # Slow decay
        
        # 5. Create new state with updated fields
        from substrate.field_types import FieldState
        new_state = FieldState(
            potential=P_new,
            actual=A_new,
            memory=M_new,
            temperature=T_new
        )
        
        # 6. PAC enforcement with Landauer costs
        new_state, pac_metrics = self.pac.enforce(new_state, correct_violations=True)
        
        # 7. Time emerges from disequilibrium
        time_rate_field, time_metrics = self.time_emer.compute_time_rate(
            new_state,
            dt_nominal=self.dt
        )
        dt_effective = self.dt * time_metrics.time_dilation_factor
        
        # 8. Update current state
        self.current_state = new_state
        
        # Update counters
        self.step_count += 1
        self.time_elapsed += dt_effective
        
        # Record state with time metrics
        recorded_state = self._record_state()
        recorded_state['time_metrics'] = {
            'disequilibrium': time_metrics.disequilibrium,
            'time_dilation': time_metrics.time_dilation_factor,
            'c_effective': time_metrics.c_effective,
            'equilibrium_approach': time_metrics.equilibrium_approach
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
            # Field statistics
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
            # Thermodynamics
            'entropy': state.entropy(),
            'free_energy': state.free_energy(),
            'disequilibrium': state.disequilibrium(),
            # SEC
            'total_heat': self.sec.total_heat_generated,
            'entropy_reduced': self.sec.total_entropy_reduced,
            'collapse_events': self.sec.collapse_event_count,
            # Confluence
            'confluence_steps': self.confluence.total_steps,
            'confluence_magnitude': self.confluence.total_confluence_magnitude,
            # Topology (use confluence to validate)
            'antiperiodic_error': self.confluence.validate_antiperiodicity(state.A)
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
