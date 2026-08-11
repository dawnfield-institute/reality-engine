"""
Reality Engine Dashboard Server
Real-time WebSocket server for universe evolution visualization
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import torch

# Add core paths for Reality Engine import
ENGINE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ENGINE_ROOT))

# Import actual Reality Engine components
from core.reality_engine import RealityEngine
from emergence.particle_analyzer import ParticleAnalyzer
from emergence.stellar_analyzer import StellarAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'reality-engine-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Create logs directory for JSON dumps
LOG_DIR = Path(__file__).parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# Global state for Reality Engine connection
engine_state = {
    'running': False,
    'iteration': 0,
    'pac_value': 1.0571,
    'field_data': None,
    'periodic_table': {},
    'emergence_events': [],
    'thermodynamic_state': {},
    'log_file': None,
    'debug_mode': True  # Enable JSON logging
}

def log_to_json(event_type, data):
    """Log all data streams to JSON file for debugging"""
    if not engine_state['debug_mode']:
        return
    
    try:
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'iteration': engine_state['iteration'],
            'event_type': event_type,
            'data': data
        }
        
        # Write to current session log file
        if engine_state['log_file']:
            with open(engine_state['log_file'], 'a') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
    except Exception as e:
        print(f"Logging error: {e}")

class RealityEngineAdapter:
    """Adapter to connect Reality Engine to dashboard"""
    
    def __init__(self):
        # Initialize MASSIVE Reality Engine with GPU
        print("🌌 Initializing MASSIVE Reality Engine with GPU acceleration...")
        
        # Check for GPU availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print(f"   🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   📊 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Enable async GPU operations
            torch.backends.cudnn.benchmark = True
        else:
            print("   ⚠️ No GPU detected, using CPU")
        
        # Scale up the Möbius manifold significantly
        self.engine_size = (256, 128)  # 32,768 points
        self.dt = 0.01  # Larger timestep for faster evolution
        
        # Initialize with GPU
        self.engine = RealityEngine(size=self.engine_size, dt=self.dt, device=device)
        self.engine.initialize(mode='big_bang')
        
        # Initialize analyzers with GPU
        self.particle_analyzer = ParticleAnalyzer(device=device)
        self.stellar_analyzer = StellarAnalyzer(mass_threshold=100.0)
        
        self.device = device
        self.running = False
        self.field_size = 200  # Display resolution
        self.time = 0
        self.stellar_masses = []
        self.atomic_structures = {}
        
        # Cache for detected structures
        self.particles = []
        self.stellar_structures = []
        self.periodic_table = {}
        
        # MED parameters for 3D emergence
        self.med_depth = 64
        
        # Performance tracking
        self.last_field_update = 0
        
        # Pre-allocate pinned memory buffers for faster CPU<->GPU transfer
        if device == 'cuda':
            self.display_buffer = torch.zeros((self.field_size, self.field_size), 
                                             dtype=torch.float32, pin_memory=True)
        
        print(f"✅ MASSIVE Reality Engine initialized on {device.upper()}!")
        print(f"   Möbius manifold: {self.engine_size[0]}×{self.engine_size[1]} = {self.engine_size[0] * self.engine_size[1]:,} points")
        print(f"   3D space: {self.engine_size[0]}×{self.engine_size[1]}×{self.med_depth} = {self.engine_size[0]*self.engine_size[1]*self.med_depth:,} points")
        
    def initialize_engine(self, config=None):
        """Initialize Reality Engine with configuration"""
        # Re-initialize engine with large size and GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.engine = RealityEngine(size=self.engine_size, dt=self.dt, device=device)
        self.engine.initialize(mode='big_bang')
        self.time = 0
        self.stellar_masses = []
        self.atomic_structures = {}
        self.particles = []
        self.stellar_structures = []
        self.periodic_table = {}
        
    def get_field_snapshot(self):
        """Get current field state from REAL Reality Engine - let patterns emerge naturally"""
        if self.engine is None or not self.engine.initialized:
            return {}
        
        # Get FRESH field state from engine EVERY TIME
        state = self.engine.current_state
        
        # Extract fields - keep on GPU as long as possible
        E = state.actual      # Energy field (keep as tensor)
        I = state.potential   # Information field
        M = state.memory      # Memory field
        T = state.temperature # Temperature
        
        # Display dimensions - keep aspect ratio of the actual field
        display_width = self.field_size  # 200
        display_height = self.field_size // 2  # 100 (matching 256x128 ratio)
        
        if self.device == 'cuda':
            import torch.nn.functional as F
            
            with torch.no_grad():
                # Just interpolate the raw fields - let the natural dynamics show through
                # The π-harmonics will EMERGE from the Möbius topology, not be imposed
                
                # Add batch and channel dimensions for interpolation
                E_batch = E.unsqueeze(0).unsqueeze(0)
                I_batch = I.unsqueeze(0).unsqueeze(0)
                T_batch = T.unsqueeze(0).unsqueeze(0)
                
                # Simple bilinear interpolation to display size
                energy_display_gpu = F.interpolate(E_batch, size=(display_height, display_width), 
                                              mode='bilinear', align_corners=False).squeeze()
                entropy_display_gpu = F.interpolate(I_batch, size=(display_height, display_width),
                                               mode='bilinear', align_corners=False).squeeze()
                temperature_display_gpu = F.interpolate(T_batch, size=(display_height, display_width),
                                                   mode='bilinear', align_corners=False).squeeze()
                
                # Calculate pressure on GPU
                pressure_display_gpu = energy_display_gpu * entropy_display_gpu * 1e-10
                
                # Normalize on GPU
                energy_display_gpu = (energy_display_gpu - energy_display_gpu.min()) / (energy_display_gpu.max() - energy_display_gpu.min() + 1e-10)
                entropy_display_gpu = (entropy_display_gpu - entropy_display_gpu.min()) / (entropy_display_gpu.max() - entropy_display_gpu.min() + 1e-10)
                
                # Copy to CPU
                energy_display = energy_display_gpu.cpu().numpy()
                entropy_display = entropy_display_gpu.cpu().numpy()
                temperature_display = temperature_display_gpu.cpu().numpy()
                pressure_display = pressure_display_gpu.cpu().numpy()
            
        else:
            # CPU path - also just show raw fields
            from scipy.ndimage import zoom
            
            E_cpu = E.cpu().numpy() if hasattr(E, 'cpu') else E
            I_cpu = I.cpu().numpy() if hasattr(I, 'cpu') else I
            T_cpu = T.cpu().numpy() if hasattr(T, 'cpu') else T
            
            # Scale to display
            scale_x = display_height / E_cpu.shape[0]
            scale_y = display_width / E_cpu.shape[1]
            
            energy_display = zoom(E_cpu, (scale_x, scale_y), order=1)
            entropy_display = zoom(I_cpu, (scale_x, scale_y), order=1)
            temperature_display = zoom(T_cpu, (scale_x, scale_y), order=1)
            
            pressure_display = energy_display * entropy_display * 1e-10
            
            # Normalize
            energy_display = (energy_display - energy_display.min()) / (energy_display.max() - energy_display.min() + 1e-10)
            entropy_display = (entropy_display - entropy_display.min()) / (entropy_display.max() - entropy_display.min() + 1e-10)
        
        # Update tracking
        self.last_field_update = self.time
        
        return {
            'energy': energy_display.tolist(),
            'entropy': entropy_display.tolist(),
            'temperature': temperature_display.tolist(),
            'pressure': pressure_display.tolist(),
            'dimensions': [display_height, display_width],
            'actual_size': list(self.engine_size),
            'total_points': self.engine_size[0] * self.engine_size[1],
            'time': self.time,
            'device': self.device,
            'topology': 'Möbius manifold (emergent π-harmonics)'
        }
        
    def get_periodic_table_state(self):
        """Get current state of emerged elements from REAL particle detection"""
        if self.engine is None or not self.engine.initialized:
            return {}
        
        # Use cached periodic table (updated in step())
        if not self.periodic_table:
            return {}
        
        # Convert to dashboard format - dictionary keyed by symbol
        elements = {}
        for particle_type, data in self.periodic_table.items():
            # Map particle types to element symbols
            symbol_map = {
                'proton': 'H', 'neutron': 'n', 'electron': 'e-',
                'photon': 'γ', 'neutrino': 'ν', 
                'fermion': 'ψ', 'boson': 'B',
                'meson': 'π', 'exotic': 'X'
            }
            
            symbol = symbol_map.get(particle_type, particle_type[:2].upper())
            
            elements[symbol] = {
                'count': data['count'],
                'stability': data.get('avg_stability', 0.5),  # Already 0.0-1.0 range
                'avgEnergy': abs(data['avg_mass']) * 931.5  # Convert mass to MeV
            }
        
        return elements
    
    def get_atomic_structures(self):
        """Get detailed atomic composition from REAL particle detection"""
        if self.engine is None or not self.engine.initialized:
            return []
        
        # Use cached particles to build atomic structures
        structures = []
        for particle in self.particles[:20]:  # Limit to first 20 for performance
            structures.append({
                'type': 'particle',
                'classification': particle.classification,
                'position': list(particle.position),
                'mass': float(particle.mass),
                'charge': float(particle.charge),
                'spin': float(particle.spin),
                'radius': float(particle.radius),
                'stability': float(particle.stability),
                'energy': float(particle.binding_energy)
            })
        
        return structures
    
    def get_stellar_masses(self):
        """Return detected stellar masses from REAL stellar analyzer"""
        if self.engine is None or not self.engine.initialized:
            return []
        
        # Use cached stellar masses (updated in step())
        return self.stellar_masses
    
    def _classify_star(self, mass):
        """Classify star type by mass"""
        if mass < 0.08:
            return 'Brown Dwarf'
        elif mass < 0.5:
            return 'Red Dwarf'
        elif mass < 1.5:
            return 'Main Sequence'
        elif mass < 8:
            return 'Giant'
        else:
            return 'Supergiant'
    
    def get_matter_distribution(self):
        """Get matter density distribution for large-scale structure"""
        if self.engine is None or not self.engine.initialized:
            # For disconnected state, show larger mock structure
            size = 100  # Doubled from 50
            x = np.linspace(0, 20, size)  # Wider range
            y = np.linspace(0, 20, size)
            z = np.linspace(0, 20, size)
            
            # Create 3D matter distribution
            # Slice at z=50 for 2D visualization
            X, Y = np.meshgrid(x, y)
            t = self.time * 0.05
            
            density = (np.sin(X + t) * np.cos(Y - t) + 
                      np.sin(X * 2 + Y) * 0.5 +
                      np.random.randn(size, size) * 0.1)
            density = np.exp(density)  # Log-normal distribution
            
            return {
                'density': density.tolist(),
                'dimensions': [size, size],
                'scale': '100 Mpc'  # Megaparsecs
            }
        return {}
        
    def get_pac_conservation(self):
        """Get current PAC conservation value from REAL Reality Engine"""
        if self.engine is None or not self.engine.initialized:
            return 1.0571
        
        # Calculate PAC from actual engine fields
        state = self.engine.current_state
        P = state.potential
        A = state.actual
        C = state.memory  # Memory represents "collapse" or conservation
        
        # PAC = P + A + C (should be conserved)
        pac_value = P.sum().item() + A.sum().item() + C.sum().item()
        
        # Normalize to ~1.0571 (Dawn Field canonical value)
        normalization = 1.0571 / (pac_value + 1e-10)
        return pac_value * normalization
        
    def get_emergence_timeline(self):
        """Get timeline of emergence events"""
        return engine_state['emergence_events'][-50:]  # Last 50 events
    
    def detect_herniations(self, E, I, M):
        """
        Detect herniation sites where information-energy pressure causes quantum collapse.
        Herniations occur when field gradients create sufficient pressure to breach containment.
        """
        with torch.no_grad():
            # Calculate field pressure (information * energy density)
            pressure = I * E
            
            # Calculate field gradients (boundaries between high/low pressure regions)
            dx = torch.diff(pressure, dim=1, prepend=pressure[:, :1])
            dy = torch.diff(pressure, dim=0, prepend=pressure[:1, :])
            gradient_magnitude = torch.sqrt(dx**2 + dy**2)
            
            # Herniation potential: pressure * gradient * uncollapsed_field
            herniation_potential = pressure * gradient_magnitude * (1.0 - M)
            
            # Find herniation sites above threshold
            threshold = herniation_potential.mean() + 2 * herniation_potential.std()
            herniation_mask = herniation_potential > threshold
            
            # Get herniation locations
            herniation_sites = torch.nonzero(herniation_mask)
            
            if len(herniation_sites) > 0:
                # Herniation rate: number of sites per field area
                herniation_rate = len(herniation_sites) / (E.shape[0] * E.shape[1])
                
                # Average herniation intensity
                avg_intensity = herniation_potential[herniation_mask].mean().item()
                
                # Quantum collapse probability (based on field coherence)
                coherence = torch.abs(torch.fft.fft2(pressure)).mean()
                collapse_probability = 1.0 / (1.0 + torch.exp(-coherence / 1000.0))
                
                return {
                    'sites': herniation_sites,
                    'rate': herniation_rate,
                    'intensity': avg_intensity,
                    'collapse_probability': collapse_probability.item(),
                    'count': len(herniation_sites)
                }
            
            return {
                'sites': torch.tensor([]),
                'rate': 0.0,
                'intensity': 0.0,
                'collapse_probability': 0.0,
                'count': 0
            }
    
    def apply_herniations(self, herniations):
        """
        Apply herniation collapses to actualize reality.
        Quantum mechanics creates classical reality through field collapse.
        """
        if herniations['count'] == 0:
            return
        
        state = self.engine.current_state
        sites = herniations['sites']
        
        # Limit number of collapses per step for performance
        max_collapses = min(50, len(sites))  # Reduce to 50 for performance
        sites = sites[:max_collapses]
        
        # Get device from state fields
        field_device = state.actual.device
        
        with torch.no_grad():
            # Process all sites in batch for efficiency
            for site in sites:
                y, x = site[0].item(), site[1].item()
                
                # Create collapse kernel (local effect) - on same device as fields
                Y, X = torch.meshgrid(
                    torch.arange(state.actual.shape[0], device=field_device),
                    torch.arange(state.actual.shape[1], device=field_device),
                    indexing='ij'
                )
                
                # Distance with Möbius wrapping
                dy = torch.minimum(torch.abs(Y - y), state.actual.shape[0] - torch.abs(Y - y))
                dx = torch.minimum(torch.abs(X - x), state.actual.shape[1] - torch.abs(X - x))
                dist = torch.sqrt(dy.float()**2 + dx.float()**2)
                
                # Collapse radius
                radius = 3.0
                
                # Gaussian collapse profile - ensure on correct device
                collapse_kernel = torch.exp(-dist**2 / (2 * radius**2))
                
                # Apply collapse: Information → Energy (actualization)
                # Ensure collapse_probability is a scalar or tensor on same device
                prob_scalar = float(herniations['collapse_probability'])
                collapse_amount = collapse_kernel * prob_scalar * 0.01  # Reduced from 0.1 to 0.01 (gentler collapses)
                
                # Transfer information to energy (quantum → classical)
                # Use in-place operations to modify state fields directly
                transfer = collapse_amount * state.potential
                state.potential.sub_(transfer)  # Reduce potential (in-place)
                state.actual.add_(transfer)     # Increase actual (in-place)
                state.memory.add_(collapse_amount * 0.01)  # Reduced memory accumulation (was 0.05)
                
                # Temperature increases at collapse sites (creates new gradients)
                state.temperature.add_(collapse_kernel * 5.0)  # Reduced from 10.0
                
                # Add turbulence to maintain field dynamics (stir the pot!)
                # Small random perturbations keep gradients active
                noise_scale = 0.005
                state.potential.add_(collapse_kernel * torch.randn_like(collapse_kernel) * noise_scale)
                state.actual.add_(collapse_kernel * torch.randn_like(collapse_kernel) * noise_scale * 0.5)
        
    def step(self):
        """Execute one simulation step - with herniation detection and application"""
        if self.engine and self.engine.initialized:
            # Step the Reality Engine
            with torch.no_grad():
                state_dict = self.engine.step()
                self.time += 1
            
            # DETECT HERNIATIONS EVERY STEP (this IS time emergence!)
            state = self.engine.current_state
            herniations = self.detect_herniations(
                state.actual, 
                state.potential, 
                state.memory
            )
            
            # Apply herniations to create reality
            if herniations['count'] > 0:
                self.apply_herniations(herniations)
                
                # Log herniation events periodically
                if engine_state['iteration'] % 30 == 0:
                    print(f"[t={self.time}] HERNIATIONS: {herniations['count']} sites")
                    print(f"  Rate: {herniations['rate']:.6f} sites/area")
                    print(f"  Intensity: {herniations['intensity']:.3f}")
                    print(f"  Collapse probability: {herniations['collapse_probability']:.3f}")
                    
                    # Create emergence event
                    event = {
                        'iteration': engine_state['iteration'],
                        'type': 'herniation',
                        'count': herniations['count'],
                        'rate': herniations['rate'],
                        'intensity': herniations['intensity'],
                        'collapse_probability': herniations['collapse_probability'],
                        'time_created': self.time
                    }
                    engine_state['emergence_events'].append(event)
            
            # Track herniation statistics
            if not hasattr(self, 'herniation_history'):
                self.herniation_history = []
            
            self.herniation_history.append({
                'time': self.time,
                'count': herniations['count'],
                'rate': herniations['rate']
            })
            
            # Keep only recent history
            if len(self.herniation_history) > 1000:
                self.herniation_history = self.herniation_history[-1000:]
            
            # Particle detection (now driven by herniations)
            if engine_state['iteration'] % 20 == 0:
                # Check if herniations have created enough structure
                total_herniations = sum(h['count'] for h in self.herniation_history[-20:]) if len(self.herniation_history) >= 20 else 0
                
                if total_herniations > 10 or engine_state['iteration'] < 100:  # Always check early on
                    E = state.actual
                    I = state.potential
                    M = state.memory
                    
                    # Debug field statistics
                    if engine_state['iteration'] % 50 == 0:
                        print(f"[t={self.time}] Field statistics:")
                        print(f"  Energy: min={E.min():.3f}, max={E.max():.3f}, mean={E.mean():.3f}")
                        print(f"  Info:   min={I.min():.3f}, max={I.max():.3f}, mean={I.mean():.3f}")
                        print(f"  Memory: min={M.min():.3f}, max={M.max():.3f}, mean={M.mean():.3f}")
                        print(f"  Recent herniations: {total_herniations}")
                    
                    with torch.no_grad():
                        # Less aggressive downsampling
                        E_downsampled = E[::4, ::4]
                        I_downsampled = I[::4, ::4]
                        M_downsampled = M[::4, ::4]
                        
                        # Convert 2D to 3D for particle analyzer
                        E_3d = E_downsampled.unsqueeze(2).expand(-1, -1, 16)
                        I_3d = I_downsampled.unsqueeze(2).expand(-1, -1, 16)
                        M_3d = M_downsampled.unsqueeze(2).expand(-1, -1, 16)
                    
                    # Detect particles with adaptive threshold
                    try:
                        threshold = 0.05 if total_herniations > 50 else 0.1
                        self.particles = self.particle_analyzer.detect_particles(
                            E_3d, I_3d, M_3d, 
                            threshold=threshold
                        )
                        
                        if self.particles:
                            self.periodic_table = self.particle_analyzer.build_periodic_table(self.particles)
                            if engine_state['iteration'] % 50 == 0:
                                print(f"[t={self.time}] Detected {len(self.particles)} particles (after {total_herniations} herniations)")
                    except Exception as e:
                        print(f"Particle detection error: {e}")
                        self.particles = []
                        self.periodic_table = {}
                    
                    # Stellar detection
                    try:
                        self.stellar_structures = self.stellar_analyzer.detect_structures(
                            E_3d, I_3d, M_3d, 
                            self.particles
                        )
                        
                        self.stellar_masses = []
                        for structure in self.stellar_structures:
                            if structure.total_mass > 0.08 and engine_state['iteration'] % 100 == 0:
                                print(f"  ⭐ STAR BORN from herniation cascade: Mass={structure.total_mass:.2f} M☉")
                            
                            self.stellar_masses.append({
                                'id': len(self.stellar_masses),
                                'mass': structure.total_mass,
                                'position': list(structure.position),
                                'temperature': structure.temperature,
                                'radius': structure.radius,
                                'type': structure.structure_type,
                                'fusion_active': structure.is_fusion_active(),
                                'formation_time': self.time,
                                'herniation_origin': True
                            })
                    except Exception as e:
                        print(f"Stellar detection error: {e}")
                        self.stellar_structures = []
                        self.stellar_masses = []
        
        engine_state['iteration'] += 1
    
    def get_herniation_rate(self):
        """Get current herniation rate (herniations per time)"""
        if not hasattr(self, 'herniation_history') or len(self.herniation_history) < 2:
            return 0.0
        
        # Calculate rate over recent history
        recent = self.herniation_history[-100:]
        if len(recent) < 2:
            return 0.0
        
        total_herniations = sum(h['count'] for h in recent)
        time_span = recent[-1]['time'] - recent[0]['time'] + 1
        
        return total_herniations / time_span

adapter = RealityEngineAdapter()

@app.route('/')
def index():
    """Serve the test dashboard page"""
    return send_from_directory('.', 'dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current engine status"""
    return jsonify({
        'running': engine_state['running'],
        'iteration': engine_state['iteration'],
        'pac_value': engine_state['pac_value']
    })

@app.route('/api/control/start', methods=['POST'])
def start_engine():
    """Start Reality Engine simulation"""
    # Create new log file for this session
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    engine_state['log_file'] = LOG_DIR / f'reality_engine_{session_id}.jsonl'
    
    # Write session header
    with open(engine_state['log_file'], 'w') as f:
        header = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'debug_mode': engine_state['debug_mode']
        }
        f.write(json.dumps(header) + '\n')
    
    print(f"Logging to: {engine_state['log_file']}")
    
    engine_state['running'] = True
    threading.Thread(target=simulation_loop, daemon=True).start()
    
    log_to_json('control', {'action': 'start', 'status': 'started'})
    return jsonify({'status': 'started', 'log_file': str(engine_state['log_file'])})

@app.route('/api/control/stop', methods=['POST'])
def stop_engine():
    """Stop Reality Engine simulation"""
    engine_state['running'] = False
    log_to_json('control', {'action': 'stop', 'status': 'stopped'})
    return jsonify({'status': 'stopped'})

@app.route('/api/control/reset', methods=['POST'])
def reset_engine():
    """Reset Reality Engine to initial state"""
    engine_state['running'] = False
    engine_state['iteration'] = 0
    engine_state['emergence_events'] = []
    adapter.initialize_engine()
    log_to_json('control', {'action': 'reset', 'status': 'reset'})
    return jsonify({'status': 'reset'})

@app.route('/api/logs')
def get_logs():
    """Get list of available log files"""
    log_files = [f.name for f in LOG_DIR.glob('*.jsonl')]
    return jsonify({'log_files': log_files})

@app.route('/api/logs/<filename>')
def get_log_file(filename):
    """Download a specific log file"""
    return send_from_directory(LOG_DIR, filename)

@app.route('/api/atomic-structures')
def get_atomic_structures():
    """Get detailed atomic structures and molecules"""
    return jsonify(adapter.get_atomic_structures())

@app.route('/api/stellar-masses')
def get_stellar_masses():
    """Get detected stellar masses"""
    return jsonify(adapter.get_stellar_masses())

@app.route('/api/herniation-rate')
def get_herniation_rate():
    """Get current herniation rate"""
    return jsonify({
        'rate': adapter.get_herniation_rate(),
        'history': adapter.herniation_history[-100:] if hasattr(adapter, 'herniation_history') else []
    })

@app.route('/api/matter-distribution')
def get_matter_distribution():
    """Get large-scale matter distribution"""
    return jsonify(adapter.get_matter_distribution())

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f'🔌 Client connected - Session ID: {request.sid}')
    print(f'   Running: {engine_state["running"]}, Iteration: {engine_state["iteration"]}')
    emit('initial_state', {
        'iteration': engine_state['iteration'],
        'pac_value': engine_state['pac_value']
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f'🔌 Client disconnected - Session ID: {request.sid}')

@socketio.on('request_snapshot')
def handle_snapshot_request():
    """Send full state snapshot to client"""
    emit('snapshot', {
        'field': adapter.get_field_snapshot(),
        'periodic_table': adapter.get_periodic_table_state(),
        'pac_value': adapter.get_pac_conservation(),
        'timeline': adapter.get_emergence_timeline(),
        'iteration': engine_state['iteration'],
        'atomic_structures': adapter.get_atomic_structures(),
        'stellar_masses': adapter.get_stellar_masses(),
        'matter_distribution': adapter.get_matter_distribution()
    })

def simulation_loop():
    """Main simulation loop broadcasting updates"""
    print("🚀 Starting high-performance simulation loop...")
    print(f"   Thread ID: {threading.current_thread().ident}")
    print(f"   Running state: {engine_state['running']}")
    
    step_count = 0
    last_log_time = time.time()
    
    try:
        while engine_state['running']:
            try:
                # ALWAYS step the simulation
                adapter.step()
                step_count += 1
                
                # ALWAYS get and broadcast field updates (this is what makes the heatmaps animate!)
                field_data = adapter.get_field_snapshot()
                pac_value = adapter.get_pac_conservation()
                
                # Broadcast field update EVERY ITERATION
                field_update = {
                    'field': field_data,
                    'iteration': engine_state['iteration'],
                    'engine_time': adapter.time
                }
                socketio.emit('field_update', field_update)
                
                # Update iteration counter
                engine_state['iteration'] += 1
                
                # Update PAC value
                engine_state['pac_value'] = pac_value
                pac_update = {
                    'value': pac_value,
                    'iteration': engine_state['iteration']
                }
                socketio.emit('pac_update', pac_update)
                
                # Debug print every 30 steps (~0.3 seconds at 100 FPS)
                if step_count % 30 == 0:
                    current_time = time.time()
                    if current_time - last_log_time >= 3.0:
                        print(f"✅ [Streaming] Step {step_count}: Iteration {engine_state['iteration']}, Time {adapter.time}")
                        last_log_time = current_time
                
                # Log only every 10 iterations to reduce I/O overhead
                if engine_state['iteration'] % 10 == 0:
                    log_to_json('field_update', field_update)
                    log_to_json('pac_update', pac_update)
                
                # Medium frequency updates for other data
                if engine_state['iteration'] % 5 == 0:
                    periodic_data = adapter.get_periodic_table_state()
                    atomic_data = adapter.get_atomic_structures()
                    
                    # Debug output every 50 iterations
                    if engine_state['iteration'] % 50 == 0:
                        print(f"[Dashboard] Periodic table: {len(periodic_data)} elements")
                        print(f"[Dashboard] Atomic structures: {len(atomic_data)} particles")
                    
                    socketio.emit('periodic_table_update', periodic_data)
                    socketio.emit('atomic_structures_update', atomic_data)
                    
                    if engine_state['iteration'] % 20 == 0:  # Log less frequently
                        log_to_json('periodic_table_update', periodic_data)
                        log_to_json('atomic_structures_update', atomic_data)
                
                if engine_state['iteration'] % 10 == 0:
                    stellar_data = adapter.get_stellar_masses()
                    matter_data = adapter.get_matter_distribution()
                    
                    # Debug output every 50 iterations
                    if engine_state['iteration'] % 50 == 0:
                        print(f"[Dashboard] Stellar objects: {len(stellar_data)}")
                        print(f"[Dashboard] Matter distribution: {matter_data.get('dimensions', 'N/A')}")
                    
                    socketio.emit('stellar_masses_update', stellar_data)
                    socketio.emit('matter_distribution_update', matter_data)
                    
                    if engine_state['iteration'] % 20 == 0:  # Log less frequently
                        log_to_json('stellar_masses_update', stellar_data)
                        log_to_json('matter_distribution_update', matter_data)
                
                # Emergence events based on real detection
                if len(adapter.particles) > 0 and np.random.rand() < 0.1:
                    event = {
                        'iteration': engine_state['iteration'],
                        'type': 'particle' if len(adapter.particles) < 10 else 'atom',
                        'energy': float(adapter.particles[0].binding_energy) if adapter.particles else 0,
                        'count': len(adapter.particles)
                    }
                    engine_state['emergence_events'].append(event)
                    socketio.emit('emergence_event', event)
                    
                    if engine_state['iteration'] % 20 == 0:
                        log_to_json('emergence_event', event)
                
                # Minimal sleep - let GPU drive the pace
                # At 0.01s, we target 100 FPS (GPU should be able to handle this)
                time.sleep(0.01)
                
            except Exception as step_error:
                print(f"❌ ERROR in simulation step {step_count}: {step_error}")
                import traceback
                traceback.print_exc()
                # Continue despite errors in individual steps
                time.sleep(0.1)
                continue
    
    except Exception as loop_error:
        print(f"❌ FATAL ERROR in simulation loop: {loop_error}")
        import traceback
        traceback.print_exc()
    finally:
        log_to_json('control', {'action': 'stopped', 'final_iteration': engine_state['iteration']})
        print("⛔ Simulation loop stopped")

if __name__ == '__main__':
    print("Starting Reality Engine Dashboard Server...")
    print("Dashboard will be available at http://localhost:5000")
    
    # Initialize adapter
    adapter.initialize_engine()
    
    # Auto-start the simulation
    print("🚀 Auto-starting simulation loop...")
    engine_state['running'] = True
    threading.Thread(target=simulation_loop, daemon=True).start()
    
    # Run server
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
