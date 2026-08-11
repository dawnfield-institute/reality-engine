import React from 'react';
import './PACMeter.css';

const PACMeter = ({ value, target = 1.0571 }) => {
  const deviation = Math.abs(value - target);
  const deviationPercent = (deviation / target) * 100;
  
  // Color coding based on deviation
  let statusColor = '#00ff88'; // Green - good
  if (deviationPercent > 0.1) statusColor = '#ffaa00'; // Orange - warning
  if (deviationPercent > 1.0) statusColor = '#ff4444'; // Red - critical
  
  const gaugePercentage = Math.min(Math.max(((value - 1.0) / 0.2) * 100, 0), 100);

  return (
    <div className="pac-meter">
      <div className="pac-display">
        <div className="pac-value" style={{ color: statusColor }}>
          {value.toFixed(6)}
        </div>
        <div className="pac-target">
          Target: {target.toFixed(4)}
        </div>
      </div>

      <div className="pac-gauge">
        <div className="gauge-track">
          <div 
            className="gauge-fill" 
            style={{ 
              width: `${gaugePercentage}%`,
              backgroundColor: statusColor
            }}
          />
          <div 
            className="gauge-target-marker" 
            style={{ left: `${((target - 1.0) / 0.2) * 100}%` }}
          />
        </div>
        <div className="gauge-labels">
          <span>1.00</span>
          <span>{target.toFixed(4)}</span>
          <span>1.20</span>
        </div>
      </div>

      <div className="pac-stats">
        <div className="stat-item">
          <span className="label">Deviation:</span>
          <span className="value" style={{ color: statusColor }}>
            {deviation.toFixed(6)}
          </span>
        </div>
        <div className="stat-item">
          <span className="label">Deviation %:</span>
          <span className="value" style={{ color: statusColor }}>
            {deviationPercent.toFixed(3)}%
          </span>
        </div>
        <div className="stat-item">
          <span className="label">Status:</span>
          <span className="value" style={{ color: statusColor }}>
            {deviationPercent < 0.1 ? 'STABLE' : deviationPercent < 1.0 ? 'WARNING' : 'CRITICAL'}
          </span>
        </div>
      </div>

      <div className="pac-explanation">
        <p className="explanation-text">
          PAC (Potential-Actualization Conservation) should remain at Xi ≈ 1.0571.
          This value represents the universal balance between potential and actualized states.
        </p>
      </div>
    </div>
  );
};

export default PACMeter;
