import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './EmergenceTimeline.css';

const EmergenceTimeline = ({ events }) => {
  if (!events || events.length === 0) {
    return (
      <div className="timeline-placeholder">
        <p>Waiting for emergence events...</p>
        <div className="pulse-indicator"></div>
      </div>
    );
  }

  // Prepare data for chart
  const chartData = events.map(event => ({
    iteration: event.iteration,
    energy: event.energy
  }));

  // Group events by type for stats
  const eventsByType = events.reduce((acc, event) => {
    acc[event.type] = (acc[event.type] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="timeline-container">
      <div className="timeline-chart">
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#0f3460" />
            <XAxis 
              dataKey="iteration" 
              stroke="#00d4ff" 
              tick={{ fill: '#a0a0a0' }}
              label={{ value: 'Iteration', position: 'insideBottom', offset: -5, fill: '#a0a0a0' }}
            />
            <YAxis 
              stroke="#00d4ff" 
              tick={{ fill: '#a0a0a0' }}
              label={{ value: 'Energy', angle: -90, position: 'insideLeft', fill: '#a0a0a0' }}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(26, 26, 46, 0.95)', 
                border: '1px solid #0f3460',
                borderRadius: '4px',
                color: '#ffffff'
              }}
            />
            <Line 
              type="monotone" 
              dataKey="energy" 
              stroke="#00ff88" 
              strokeWidth={2}
              dot={{ fill: '#00ff88', r: 3 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="event-stats">
        <h3>Event Types</h3>
        <div className="event-type-list">
          {Object.entries(eventsByType).map(([type, count]) => (
            <div key={type} className="event-type-item">
              <span className={`event-icon ${type}`}>
                {type === 'particle' ? '●' : type === 'atom' ? '⚛' : '⬡'}
              </span>
              <span className="event-type-name">{type}</span>
              <span className="event-type-count">{count}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="recent-events">
        <h3>Recent Events</h3>
        <div className="events-list">
          {events.slice(-10).reverse().map((event, idx) => (
            <div key={idx} className="event-item">
              <span className={`event-marker ${event.type}`}></span>
              <div className="event-details">
                <div className="event-header">
                  <span className="event-type">{event.type.toUpperCase()}</span>
                  <span className="event-iteration">@{event.iteration.toLocaleString()}</span>
                </div>
                <div className="event-energy">Energy: {event.energy.toFixed(2)}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default EmergenceTimeline;
