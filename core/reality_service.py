"""
Reality Engine Service - Main Runtime Service

This is the core service that runs the Reality Engine independently of any UI.
It manages the engine lifecycle, processes herniations, detects particles,
and provides a clean API for clients (like dashboards) to subscribe to updates.

The service runs in its own thread/process and can be accessed by multiple clients.
"""

import time
import threading
import torch
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
ENGINE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ENGINE_ROOT))

from core.reality_engine import RealityEngine
from emergence.particle_analyzer import ParticleAnalyzer
from emergence.stellar_analyzer import StellarAnalyzer
from emergence.herniation_detector import HerniationDetector


@dataclass
class EngineConfig:
    """Configuration for Reality Engine Service"""
    size: tuple = (256, 128)
    dt: float = 0.01
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    med_depth: int = 64
    
    # Herniation parameters
    herniation_threshold: float = 2.0
    collapse_strength: float = 0.001  # Reduced 10x to preserve PAC
    memory_rate: float = 0.001  # Reduced to match
    turbulence_scale: float = 0.0005  # Reduced proportionally
    
    # Detection intervals
    particle_detection_interval: int = 20
    stellar_detection_interval: int = 40
    
    # Performance
    target_fps: float = 100.0
    update_throttle: float = 0.01  # Min seconds between updates


class RealityEngineService:
    """
    Main service that runs the Reality Engine and manages its lifecycle.
    
    This is the "real thing" - the actual engine that computes reality.
    UIs and other tools connect to this service as clients.
    """
    
    def __init__(self, config: Optional[EngineConfig] = None):
        """Initialize the Reality Engine Service"""
        self.config = config or EngineConfig()
        self.device = self._setup_device()
        
        print(f"🌌 Initializing Reality Engine Service...")
        print(f"   Device: {self.device}")
        print(f"   Size: {self.config.size[0]}×{self.config.size[1]}")
        
        # Core components (initialized on start)
        self.engine: Optional[RealityEngine] = None
        self.particle_analyzer: Optional[ParticleAnalyzer] = None
        self.stellar_analyzer: Optional[StellarAnalyzer] = None
        self.herniation_detector: Optional[HerniationDetector] = None
        
        # State tracking
        self.running = False
        self.iteration = 0
        self.time = 0
        self.initialized = False
        
        # Data cache for clients
        self.particles = []
        self.stellar_structures = []
        self.periodic_table = {}
        self.herniation_stats = {}
        
        # Simulation thread
        self._sim_thread: Optional[threading.Thread] = None
        self._update_callbacks: List[Callable] = []
        
        # Performance tracking
        self._last_update_time = 0
        self._step_count = 0
        self._start_time = 0
        
    def _setup_device(self) -> str:
        """Setup compute device"""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"   🚀 GPU detected: {gpu_name}")
                print(f"   📊 GPU memory: {gpu_memory:.1f} GB")
                return device
            return 'cpu'
        return self.config.device
    
    def initialize(self, mode='big_bang'):
        """
        Initialize the engine and all analyzers.
        
        Args:
            mode: Initialization mode ('big_bang', 'vacuum', 'custom')
        """
        if self.initialized:
            print("⚠️  Service already initialized")
            return
        
        print(f"🔧 Initializing Reality Engine components...")
        
        # Initialize core engine
        self.engine = RealityEngine(
            size=self.config.size,
            dt=self.config.dt,
            device=self.device
        )
        # Set MED depth separately if needed
        if hasattr(self.engine, 'med_depth'):
            self.engine.med_depth = self.config.med_depth
        
        self.engine.initialize(mode=mode)
        
        # Initialize analyzers
        self.particle_analyzer = ParticleAnalyzer(device=self.device)
        self.stellar_analyzer = StellarAnalyzer(mass_threshold=100.0)
        
        # Initialize herniation detector
        self.herniation_detector = HerniationDetector(
            device=self.device,
            threshold_sigma=self.config.herniation_threshold,
            collapse_strength=self.config.collapse_strength,
            memory_rate=self.config.memory_rate,
            turbulence_scale=self.config.turbulence_scale
        )
        
        # Initialize Analog Architecture Extension
        # This wraps the engine and adds multi-scale capability
        try:
            from core.reality_engine import AnalogExtension
            self.analog = AnalogExtension(
                base_engine=self.engine,
                enable_spawning=True,
                spawn_threshold=0.5,  # Spawn at moderate herniation intensity
                max_centers=10
            )
            self.analog_enabled = True
            print("   🔬 Analog architecture extension enabled")
        except Exception as e:
            print(f"   ⚠️  Analog extension disabled: {e}")
            self.analog = None
            self.analog_enabled = False
        
        self.initialized = True
        print(f"✅ Reality Engine Service initialized!")
        print(f"   Möbius manifold: {self.config.size[0]}×{self.config.size[1]} = {self.config.size[0]*self.config.size[1]:,} points")
        print(f"   3D space: {self.config.size[0]}×{self.config.size[1]}×{self.config.med_depth} = {self.config.size[0]*self.config.size[1]*self.config.med_depth:,} points")
        
    def start(self):
        """Start the engine service in a background thread"""
        if not self.initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        if self.running:
            print("⚠️  Service already running")
            return
        
        print("🚀 Starting Reality Engine Service...")
        self.running = True
        self._start_time = time.time()
        self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()
        print("✅ Service started")
    
    def stop(self):
        """Stop the engine service"""
        if not self.running:
            return
        
        print("⏹️  Stopping Reality Engine Service...")
        self.running = False
        if self._sim_thread:
            self._sim_thread.join(timeout=5.0)
        print("✅ Service stopped")
    
    def _simulation_loop(self):
        """Main simulation loop - runs in background thread"""
        print("🔄 Simulation loop started")
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Step the engine
                self._step_engine()
                
                # Notify subscribers (with throttling)
                current_time = time.time()
                if current_time - self._last_update_time >= self.config.update_throttle:
                    self._notify_subscribers()
                    self._last_update_time = current_time
                
                # Maintain target FPS
                elapsed = time.time() - loop_start
                target_sleep = (1.0 / self.config.target_fps) - elapsed
                if target_sleep > 0:
                    time.sleep(target_sleep)
                    
        except Exception as e:
            print(f"❌ ERROR in simulation loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("⛔ Simulation loop stopped")
    
    def _step_engine(self):
        """Execute one simulation step with all detection and processing"""
        with torch.no_grad():
            # Step Reality Engine
            self.engine.step()
            self.iteration += 1
            self.time += 1
            self._step_count += 1
            
            # Get current state
            state = self.engine.current_state
            
            # Detect and apply herniations EVERY step
            herniations = self.herniation_detector.detect(
                state.actual,
                state.potential,
                state.memory
            )
            
            if herniations['count'] > 0:
                self.herniation_detector.apply(herniations, state)
                
                # Pass herniation data to analog extension for spawning
                if self.analog_enabled and self.analog:
                    try:
                        self.analog.step(herniation_data=herniations)
                    except Exception as e:
                        print(f"⚠️  Analog step error: {e}")
                
                # Log periodically
                if self.iteration % 30 == 0:
                    print(f"[t={self.iteration}] HERNIATIONS: {herniations['count']} sites")
                    print(f"  Rate: {herniations['rate']:.6f} sites/area")
                    print(f"  Intensity: {herniations['intensity']:.3f}")
                    print(f"  Collapse probability: {herniations['collapse_probability']:.3f}")
            
            # Periodic particle detection
            if self.iteration % self.config.particle_detection_interval == 0:
                self._detect_particles()
            
            # Periodic stellar detection
            if self.iteration % self.config.stellar_detection_interval == 0:
                self._detect_stellar_structures()
    
    def _detect_particles(self):
        """Run particle detection on 3D actualized fields"""
        # TODO: Rewrite particle detection to be pure torch
        # For now, skip to test analog architecture
        self.particles = []
        self.periodic_table = {}
        return
    
    def _detect_stellar_structures(self):
        """Run stellar structure detection"""
        if not self.particles:
            return
        
        # Detect stellar structures from particles
        self.stellar_structures = self.stellar_analyzer.detect_structures(
            self.particles
        )
        
        if self.stellar_structures and self.iteration % 100 == 0:
            total_mass = sum(s.mass for s in self.stellar_structures)
            print(f"  Detected {len(self.stellar_structures)} stellar structures")
            print(f"  Total mass: {total_mass:.1f} M☉")
    
    def _notify_subscribers(self):
        """Notify all registered callbacks of state update"""
        if not self._update_callbacks:
            return
        
        # Prepare update data
        update = self.get_current_state()
        
        # Call all subscribers
        for callback in self._update_callbacks:
            try:
                callback(update)
            except Exception as e:
                print(f"⚠️  Error in subscriber callback: {e}")
    
    def subscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to engine state updates.
        
        Args:
            callback: Function that receives state dictionary on each update
        """
        self._update_callbacks.append(callback)
        print(f"✅ Subscriber added (total: {len(self._update_callbacks)})")
    
    def unsubscribe(self, callback: Callable):
        """Remove a subscription callback"""
        if callback in self._update_callbacks:
            self._update_callbacks.remove(callback)
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current engine state snapshot.
        
        Returns complete state dictionary for clients.
        """
        if not self.initialized or not self.engine:
            return {
                'initialized': False,
                'iteration': 0
            }
        
        state = self.engine.current_state
        
        # Calculate PAC conservation
        pac_value = self._calculate_pac()
        
        # Get herniation statistics
        herniation_stats = self.herniation_detector.get_statistics()
        
        # Calculate performance metrics
        elapsed = time.time() - self._start_time if self._start_time else 1.0
        fps = self._step_count / elapsed if elapsed > 0 else 0
        
        # Get analog statistics if enabled
        analog_stats = None
        if self.analog_enabled and self.analog:
            try:
                analog_stats = self.analog.get_statistics()
            except Exception as e:
                print(f"⚠️  Error getting analog stats: {e}")
        
        return {
            'initialized': True,
            'running': self.running,
            'iteration': self.iteration,
            'time': self.time,
            'device': self.device,
            'pac_value': pac_value,
            'particle_count': len(self.particles),
            'stellar_count': len(self.stellar_structures),
            'herniation_stats': herniation_stats,
            'periodic_table': self._format_periodic_table(),
            'analog_stats': analog_stats,  # NEW: Analog architecture statistics
            'performance': {
                'fps': fps,
                'elapsed': elapsed,
                'step_count': self._step_count
            }
        }
    
    def get_field_snapshot(self, display_size: tuple = (200, 100)) -> Dict[str, Any]:
        """
        Get field visualization data.
        
        Args:
            display_size: (height, width) for display resolution
            
        Returns:
            Dictionary with energy, entropy, temperature, pressure arrays
        """
        if not self.initialized or not self.engine:
            return {}
        
        state = self.engine.current_state
        display_height, display_width = display_size
        
        # Use GPU for interpolation if available
        if self.device == 'cuda':
            import torch.nn.functional as F
            
            with torch.no_grad():
                # Add batch and channel dimensions
                E_batch = state.actual.unsqueeze(0).unsqueeze(0)
                I_batch = state.potential.unsqueeze(0).unsqueeze(0)
                T_batch = state.temperature.unsqueeze(0).unsqueeze(0)
                
                # Interpolate to display size
                energy_gpu = F.interpolate(
                    E_batch, size=(display_height, display_width),
                    mode='bilinear', align_corners=False
                ).squeeze()
                
                entropy_gpu = F.interpolate(
                    I_batch, size=(display_height, display_width),
                    mode='bilinear', align_corners=False
                ).squeeze()
                
                temperature_gpu = F.interpolate(
                    T_batch, size=(display_height, display_width),
                    mode='bilinear', align_corners=False
                ).squeeze()
                
                # Calculate pressure
                pressure_gpu = energy_gpu * entropy_gpu * 1e-10
                
                # Normalize
                energy_gpu = (energy_gpu - energy_gpu.min()) / (energy_gpu.max() - energy_gpu.min() + 1e-10)
                entropy_gpu = (entropy_gpu - entropy_gpu.min()) / (entropy_gpu.max() - entropy_gpu.min() + 1e-10)
                
                # Copy to CPU
                energy = energy_gpu.cpu().numpy()
                entropy = entropy_gpu.cpu().numpy()
                temperature = temperature_gpu.cpu().numpy()
                pressure = pressure_gpu.cpu().numpy()
        else:
            # CPU path
            from scipy.ndimage import zoom
            
            E = state.actual.cpu().numpy() if hasattr(state.actual, 'cpu') else state.actual
            I = state.potential.cpu().numpy() if hasattr(state.potential, 'cpu') else state.potential
            T = state.temperature.cpu().numpy() if hasattr(state.temperature, 'cpu') else state.temperature
            
            scale_y = display_height / E.shape[0]
            scale_x = display_width / E.shape[1]
            
            energy = zoom(E, (scale_y, scale_x), order=1)
            entropy = zoom(I, (scale_y, scale_x), order=1)
            temperature = zoom(T, (scale_y, scale_x), order=1)
            
            pressure = energy * entropy * 1e-10
            
            # Normalize
            energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-10)
            entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-10)
        
        return {
            'energy': energy.tolist(),
            'entropy': entropy.tolist(),
            'temperature': temperature.tolist(),
            'pressure': pressure.tolist(),
            'dimensions': [display_height, display_width],
            'actual_size': list(self.config.size),
            'total_points': self.config.size[0] * self.config.size[1],
            'time': self.time,
            'device': self.device,
            'topology': 'Möbius manifold (emergent π-harmonics)'
        }
    
    def _calculate_pac(self) -> float:
        """Calculate PAC (Particle-Antiparticle Conservation) value"""
        if not self.engine or not self.engine.initialized:
            return 1.0571
        
        state = self.engine.current_state
        
        # PAC = (total_potential + total_actual + total_memory) / initial_value
        total_potential = torch.sum(torch.abs(state.potential)).item()
        total_actual = torch.sum(torch.abs(state.actual)).item()
        total_memory = torch.sum(state.memory).item()
        
        total = total_potential + total_actual + total_memory
        
        # Normalize to initial PAC value
        expected_total = self.config.size[0] * self.config.size[1] * 1.0571
        
        return total / expected_total if expected_total > 0 else 1.0571
    
    def _format_periodic_table(self) -> Dict[str, Any]:
        """Format periodic table for client consumption"""
        if not self.periodic_table:
            return {}
        
        # Map particle types to element symbols
        symbol_map = {
            'proton': 'H', 'neutron': 'n', 'electron': 'e-',
            'photon': 'γ', 'neutrino': 'ν',
            'fermion': 'ψ', 'boson': 'B',
            'meson': 'π', 'exotic': 'X'
        }
        
        elements = {}
        for particle_type, data in self.periodic_table.items():
            symbol = symbol_map.get(particle_type, particle_type[:2].upper())
            
            elements[symbol] = {
                'count': data['count'],
                'stability': data.get('avg_stability', 0.5),
                'avgEnergy': abs(data['avg_mass']) * 931.5  # Convert mass to MeV
            }
        
        return elements
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        elapsed = time.time() - self._start_time if self._start_time else 1.0
        fps = self._step_count / elapsed if elapsed > 0 else 0
        
        return {
            'fps': fps,
            'elapsed_time': elapsed,
            'total_steps': self._step_count,
            'total_iterations': self.iteration,
            'subscribers': len(self._update_callbacks),
            'device': self.device,
            'herniation_stats': self.herniation_detector.get_statistics()
        }


# Global service instance
_service_instance: Optional[RealityEngineService] = None


def get_service(config: Optional[EngineConfig] = None) -> RealityEngineService:
    """
    Get or create the global Reality Engine Service instance.
    
    This ensures only one service runs at a time.
    """
    global _service_instance
    
    if _service_instance is None:
        _service_instance = RealityEngineService(config)
    
    return _service_instance
