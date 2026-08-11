import React from 'react';
import Plot from 'react-plotly.js';
import './UniverseHeatmap.css';

const UniverseHeatmap = ({ data }) => {
  if (!data || !data.energy) {
    return (
      <div className="heatmap-placeholder">
        <p>Waiting for field data...</p>
        <div className="loading-spinner"></div>
      </div>
    );
  }

  const { energy, entropy, dimensions } = data;

  return (
    <div className="heatmap-container">
      <div className="heatmap-view">
        <h3>Energy Field</h3>
        <Plot
          data={[
            {
              z: energy,
              type: 'heatmap',
              colorscale: 'Hot',
              showscale: true,
              colorbar: {
                title: 'Energy',
                titleside: 'right'
              }
            }
          ]}
          layout={{
            autosize: true,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#ffffff' },
            xaxis: { title: 'X', color: '#00d4ff' },
            yaxis: { title: 'Y', color: '#00d4ff' },
            margin: { l: 50, r: 50, t: 30, b: 50 }
          }}
          config={{ responsive: true }}
          style={{ width: '100%', height: '350px' }}
        />
      </div>

      <div className="heatmap-view">
        <h3>Entropy Field</h3>
        <Plot
          data={[
            {
              z: entropy,
              type: 'heatmap',
              colorscale: 'Viridis',
              showscale: true,
              colorbar: {
                title: 'Entropy',
                titleside: 'right'
              }
            }
          ]}
          layout={{
            autosize: true,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#ffffff' },
            xaxis: { title: 'X', color: '#00d4ff' },
            yaxis: { title: 'Y', color: '#00d4ff' },
            margin: { l: 50, r: 50, t: 30, b: 50 }
          }}
          config={{ responsive: true }}
          style={{ width: '100%', height: '350px' }}
        />
      </div>

      <div className="field-stats">
        <div className="stat">
          <span className="stat-label">Dimensions:</span>
          <span className="stat-value">{dimensions[0]} × {dimensions[1]}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Avg Energy:</span>
          <span className="stat-value">
            {(energy.flat().reduce((a, b) => a + b, 0) / (dimensions[0] * dimensions[1])).toFixed(4)}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Avg Entropy:</span>
          <span className="stat-value">
            {(entropy.flat().reduce((a, b) => a + b, 0) / (dimensions[0] * dimensions[1])).toFixed(4)}
          </span>
        </div>
      </div>
    </div>
  );
};

export default UniverseHeatmap;
