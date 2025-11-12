# Reality Engine Dashboard

A real-time visualization dashboard for monitoring universe evolution in the Reality Engine.

## Features

- **Universe Field Evolution**: Live heatmaps showing energy and entropy distributions
- **PAC Conservation Monitor**: Real-time tracking of Potential-Actualization Conservation (Xi ≈ 1.0571)
- **Periodic Table**: Dynamic visualization of emerged elements with stability indicators
- **Emergence Timeline**: Timeline of particle, atom, and molecule formation events
- **Control Panel**: Start/stop/reset simulation controls

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

1. Install Python dependencies:
```bash
pip install -r ../requirements.txt
```

2. Install frontend dependencies:
```bash
cd frontend
npm install
```

### Running the Dashboard

#### Option 1: Automated Start (Windows PowerShell)

From the `reality-engine` root directory:
```powershell
.\start-dashboard.ps1
```

#### Option 2: Manual Start

Terminal 1 - Backend:
```bash
python dashboard/server.py
```

Terminal 2 - Frontend:
```bash
cd dashboard/frontend
npm start
```

The dashboard will be available at `http://localhost:3000`

## Architecture

### Backend (`server.py`)
- Flask server with WebSocket support (Socket.IO)
- REST API for control operations
- Real-time broadcast of field updates, PAC values, and emergence events
- Adapter pattern for Reality Engine integration

### Frontend (React)
- Real-time WebSocket connection for live updates
- Plotly.js for interactive heatmaps
- Recharts for timeline visualization
- Custom periodic table component
- Responsive grid layout

## Integration with Reality Engine

The dashboard uses `RealityEngineAdapter` to connect to the core simulation. To integrate with your actual Reality Engine:

1. Import your Reality Engine in `server.py`:
```python
from core.field_engine import RealityEngine
```

2. Update `RealityEngineAdapter.initialize_engine()` to initialize your engine

3. Update snapshot methods to extract data from your engine:
   - `get_field_snapshot()` - Extract energy/entropy field data
   - `get_periodic_table_state()` - Extract emerged elements
   - `get_pac_conservation()` - Calculate PAC value
   - `get_emergence_timeline()` - Extract emergence events

## API Endpoints

### REST API

- `GET /api/status` - Get current simulation status
- `POST /api/control/start` - Start simulation
- `POST /api/control/stop` - Stop simulation
- `POST /api/control/reset` - Reset simulation to initial state

### WebSocket Events

**Client → Server:**
- `request_snapshot` - Request full state snapshot

**Server → Client:**
- `initial_state` - Initial connection data
- `field_update` - Field energy/entropy data (every 10 iterations)
- `periodic_table_update` - Element emergence data (every 50 iterations)
- `pac_update` - PAC conservation value (every iteration)
- `emergence_event` - New particle/atom/molecule detected
- `snapshot` - Complete state snapshot

## Customization

### Adding New Visualizations

1. Create component in `frontend/src/components/`
2. Import and add to `App.js` dashboard grid
3. Subscribe to relevant WebSocket events
4. Add backend data source in `server.py` if needed

### Adjusting Update Frequencies

In `server.py`, modify the `simulation_loop()` function:
```python
if engine_state['iteration'] % N == 0:  # Update every N iterations
    socketio.emit('your_event', data)
```

## Development Notes

Currently using mock data for initial development. Connect to actual Reality Engine by implementing the TODOs in `RealityEngineAdapter`.

## License

Part of the Dawn Field Institute Reality Engine project.
