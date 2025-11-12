import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import UniverseHeatmap from './components/UniverseHeatmap';
import PeriodicTable from './components/PeriodicTable';
import PACMeter from './components/PACMeter';
import EmergenceTimeline from './components/EmergenceTimeline';
import ControlPanel from './components/ControlPanel';
import './App.css';

const socket = io('http://localhost:5000');

function App() {
  const [connected, setConnected] = useState(false);
  const [iteration, setIteration] = useState(0);
  const [pacValue, setPacValue] = useState(1.0571);
  const [fieldData, setFieldData] = useState(null);
  const [periodicTable, setPeriodicTable] = useState({});
  const [emergenceEvents, setEmergenceEvents] = useState([]);
  const [running, setRunning] = useState(false);
  const [stellarMasses, setStellarMasses] = useState([]);
  const [matterDistribution, setMatterDistribution] = useState(null);
  const [atomicStructures, setAtomicStructures] = useState([]);

  useEffect(() => {
    // Socket connection handlers
    socket.on('connect', () => {
      console.log('Connected to Reality Engine');
      setConnected(true);
      socket.emit('request_snapshot');
    });

    socket.on('disconnect', () => {
      console.log('Disconnected from Reality Engine');
      setConnected(false);
    });

    socket.on('initial_state', (data) => {
      setIteration(data.iteration);
      setPacValue(data.pac_value);
    });

    socket.on('field_update', (data) => {
      setFieldData(data.field);
      setIteration(data.iteration);
    });

    socket.on('periodic_table_update', (data) => {
      setPeriodicTable(data);
    });

    socket.on('stellar_masses_update', (data) => {
      setStellarMasses(data);
    });

    socket.on('matter_distribution_update', (data) => {
      setMatterDistribution(data);
    });

    socket.on('atomic_structures_update', (data) => {
      setAtomicStructures(data);
    });

    socket.on('pac_update', (data) => {
      setPacValue(data.value);
      setIteration(data.iteration);
    });

    socket.on('emergence_event', (event) => {
      setEmergenceEvents(prev => [...prev.slice(-49), event]);
    });

    socket.on('snapshot', (data) => {
      setFieldData(data.field);
      setPeriodicTable(data.periodic_table);
      setPacValue(data.pac_value);
      setEmergenceEvents(data.timeline);
      setIteration(data.iteration);
    });

    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('initial_state');
      socket.off('field_update');
      socket.off('periodic_table_update');
      socket.off('stellar_masses_update');
      socket.off('matter_distribution_update');
      socket.off('atomic_structures_update');
      socket.off('pac_update');
      socket.off('emergence_event');
      socket.off('snapshot');
    };
  }, []);

  const handleStart = async () => {
    const response = await fetch('/api/control/start', { method: 'POST' });
    if (response.ok) {
      setRunning(true);
    }
  };

  const handleStop = async () => {
    const response = await fetch('/api/control/stop', { method: 'POST' });
    if (response.ok) {
      setRunning(false);
    }
  };

  const handleReset = async () => {
    const response = await fetch('/api/control/reset', { method: 'POST' });
    if (response.ok) {
      setRunning(false);
      setIteration(0);
      setEmergenceEvents([]);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Reality Engine Dashboard</h1>
        <div className="status-bar">
          <span className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
            {connected ? '● Connected' : '○ Disconnected'}
          </span>
          <span className="iteration">Iteration: {iteration.toLocaleString()}</span>
        </div>
      </header>

      <ControlPanel 
        running={running}
        onStart={handleStart}
        onStop={handleStop}
        onReset={handleReset}
      />

      <div className="dashboard-grid">
        <div className="panel universe-panel">
          <h2>Universe Field Evolution</h2>
          <UniverseHeatmap data={fieldData} />
        </div>

        <div className="panel pac-panel">
          <h2>PAC Conservation</h2>
          <PACMeter value={pacValue} target={1.0571} />
        </div>

        <div className="panel periodic-panel">
          <h2>Periodic Table (Emerged Elements)</h2>
          <PeriodicTable elements={periodicTable} />
        </div>

        <div className="panel timeline-panel">
          <h2>Emergence Timeline</h2>
          <EmergenceTimeline events={emergenceEvents} />
        </div>
      </div>

      <footer className="App-footer">
        <p>Dawn Field Institute • Reality Engine v0.1.0-alpha</p>
      </footer>
    </div>
  );
}

export default App;
