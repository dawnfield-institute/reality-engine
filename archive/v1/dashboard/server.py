"""
Reality Engine Dashboard Server - Web API Only

This is a thin web server layer that connects the dashboard UI
to the Reality Engine Service. It handles HTTP and WebSocket routing,
but contains no engine logic itself.

The engine runs independently via RealityEngineService.
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
from datetime import datetime
from pathlib import Path

from engine_client import RealityEngineClient, EngineConfig

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'reality-engine-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Create logs directory
LOG_DIR = Path(__file__).parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# Initialize Reality Engine Client
print("🔌 Connecting to Reality Engine Service...")
engine_client = RealityEngineClient(EngineConfig(
    size=(256, 128),
    dt=0.01,
    device='auto',
    med_depth=64
))

# Initialize and auto-start the engine
engine_client.initialize(mode='big_bang')
engine_client.start()

print("✅ Dashboard connected to Reality Engine Service")


def broadcast_update(state: dict):
    """
    Callback for engine updates - broadcasts to all WebSocket clients.
    
    This is called by the engine service whenever state changes.
    """
    try:
        # Broadcast field update
        if 'initialized' in state and state['initialized']:
            # Get field snapshot
            field_data = engine_client.get_field_snapshot(display_size=(200, 100))
            
            socketio.emit('field_update', {
                'field': field_data,
                'iteration': state.get('iteration', 0),
                'engine_time': state.get('time', 0)
            })
            
            # Broadcast PAC update
            socketio.emit('pac_update', {
                'value': state.get('pac_value', 1.0571),
                'iteration': state.get('iteration', 0)
            })
            
            # Broadcast periodic table (every 5 iterations)
            if state.get('iteration', 0) % 5 == 0:
                periodic_table = engine_client.get_periodic_table()
                socketio.emit('periodic_table_update', periodic_table)
                
                # Atomic structures
                particles = engine_client.get_particles()
                atomic_structures = []
                for particle in particles[:20]:  # Limit to 20
                    atomic_structures.append({
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
                socketio.emit('atomic_structures_update', atomic_structures)
            
            # Broadcast stellar masses (every 10 iterations)
            if state.get('iteration', 0) % 10 == 0:
                stellar_structures = engine_client.get_stellar_structures()
                stellar_data = []
                for structure in stellar_structures[:20]:
                    stellar_data.append({
                        'mass': float(structure.mass),
                        'position': list(structure.center_of_mass),
                        'radius': float(structure.radius),
                        'luminosity': float(structure.luminosity),
                        'particle_count': structure.particle_count
                    })
                socketio.emit('stellar_masses_update', stellar_data)
                
    except Exception as e:
        print(f"⚠️  Error in broadcast_update: {e}")


# Subscribe to engine updates
engine_client.subscribe_updates(broadcast_update)


# ============================================================================
# HTTP REST API Routes
# ============================================================================

@app.route('/')
def index():
    """Serve main dashboard page"""
    return send_from_directory('.', 'dashboard.html')


@app.route('/api/status')
def get_status():
    """Get engine status"""
    status = engine_client.get_status()
    return jsonify(status)


@app.route('/api/control/start', methods=['POST'])
def start_engine():
    """Start Reality Engine simulation"""
    if not engine_client.is_initialized():
        engine_client.initialize()
    
    engine_client.start()
    return jsonify({'status': 'started'})


@app.route('/api/control/stop', methods=['POST'])
def stop_engine():
    """Stop Reality Engine simulation"""
    engine_client.stop()
    return jsonify({'status': 'stopped'})


@app.route('/api/control/reset', methods=['POST'])
def reset_engine():
    """Reset Reality Engine to initial state"""
    engine_client.stop()
    engine_client.initialize(mode='big_bang')
    return jsonify({'status': 'reset'})


@app.route('/api/logs')
def list_logs():
    """List available log files"""
    logs = [f.name for f in LOG_DIR.glob('*.jsonl')]
    return jsonify({'logs': logs})


@app.route('/api/logs/<filename>')
def get_log(filename):
    """Get specific log file"""
    return send_from_directory(LOG_DIR, filename)


@app.route('/api/atomic-structures')
def get_atomic_structures():
    """Get detailed atomic composition"""
    particles = engine_client.get_particles()
    structures = []
    for particle in particles[:20]:
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
    return jsonify(structures)


@app.route('/api/stellar-masses')
def get_stellar_masses():
    """Get stellar structure data"""
    structures = engine_client.get_stellar_structures()
    stellar_data = []
    for structure in structures[:20]:
        stellar_data.append({
            'mass': float(structure.mass),
            'position': list(structure.center_of_mass),
            'radius': float(structure.radius),
            'luminosity': float(structure.luminosity),
            'particle_count': structure.particle_count
        })
    return jsonify(stellar_data)


@app.route('/api/herniation-rate')
def get_herniation_rate():
    """Get herniation statistics"""
    stats = engine_client.get_herniation_stats()
    return jsonify(stats)


@app.route('/api/matter-distribution')
def get_matter_distribution():
    """Get large-scale matter distribution"""
    # Simplified - just return particle density
    particles = engine_client.get_particles()
    return jsonify({
        'particle_count': len(particles),
        'stellar_count': len(engine_client.get_stellar_structures())
    })


@app.route('/api/performance')
def get_performance():
    """Get performance statistics"""
    stats = engine_client.get_performance_stats()
    return jsonify(stats)


# ============================================================================
# WebSocket Event Handlers
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f'🔌 Client connected - Session ID: {request.sid}')
    
    # Send initial state
    status = engine_client.get_status()
    emit('initial_state', {
        'iteration': status.get('iteration', 0),
        'pac_value': status.get('pac_value', 1.0571),
        'running': engine_client.is_running()
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f'🔌 Client disconnected - Session ID: {request.sid}')


@socketio.on('request_snapshot')
def handle_snapshot_request():
    """Send full state snapshot to client"""
    status = engine_client.get_status()
    field_data = engine_client.get_field_snapshot()
    periodic_table = engine_client.get_periodic_table()
    
    particles = engine_client.get_particles()
    atomic_structures = []
    for particle in particles[:20]:
        atomic_structures.append({
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
    
    emit('snapshot', {
        'field': field_data,
        'periodic_table': periodic_table,
        'pac_value': status.get('pac_value', 1.0571),
        'iteration': status.get('iteration', 0),
        'atomic_structures': atomic_structures,
        'running': engine_client.is_running()
    })


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    print("🚀 Starting Reality Engine Dashboard Server...")
    print("   Dashboard will be available at http://localhost:5000")
    print(f"   Engine running: {engine_client.is_running()}")
    print(f"   Engine iteration: {engine_client.iteration}")
    
    # Run server
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
