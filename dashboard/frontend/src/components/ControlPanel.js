import React from 'react';
import './ControlPanel.css';

const ControlPanel = ({ running, onStart, onStop, onReset }) => {
  return (
    <div className="control-panel">
      <div className="control-buttons">
        <button 
          className="control-btn start-btn"
          onClick={onStart}
          disabled={running}
        >
          ▶ Start
        </button>
        <button 
          className="control-btn stop-btn"
          onClick={onStop}
          disabled={!running}
        >
          ■ Stop
        </button>
        <button 
          className="control-btn reset-btn"
          onClick={onReset}
        >
          ↻ Reset
        </button>
      </div>
      
      <div className="status-indicator">
        <span className={`status-light ${running ? 'running' : 'stopped'}`}></span>
        <span className="status-text">
          {running ? 'SIMULATION RUNNING' : 'SIMULATION STOPPED'}
        </span>
      </div>
    </div>
  );
};

export default ControlPanel;
