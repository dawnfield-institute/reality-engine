import React from 'react';
import './PeriodicTable.css';

// Basic periodic table structure - first 18 elements for initial display
const ELEMENT_INFO = {
  'H': { number: 1, name: 'Hydrogen', group: 'nonmetal' },
  'He': { number: 2, name: 'Helium', group: 'noble-gas' },
  'Li': { number: 3, name: 'Lithium', group: 'alkali-metal' },
  'Be': { number: 4, name: 'Beryllium', group: 'alkaline-earth' },
  'B': { number: 5, name: 'Boron', group: 'metalloid' },
  'C': { number: 6, name: 'Carbon', group: 'nonmetal' },
  'N': { number: 7, name: 'Nitrogen', group: 'nonmetal' },
  'O': { number: 8, name: 'Oxygen', group: 'nonmetal' },
  'F': { number: 9, name: 'Fluorine', group: 'halogen' },
  'Ne': { number: 10, name: 'Neon', group: 'noble-gas' },
  'Na': { number: 11, name: 'Sodium', group: 'alkali-metal' },
  'Mg': { number: 12, name: 'Magnesium', group: 'alkaline-earth' },
  'Al': { number: 13, name: 'Aluminum', group: 'post-transition' },
  'Si': { number: 14, name: 'Silicon', group: 'metalloid' },
  'P': { number: 15, name: 'Phosphorus', group: 'nonmetal' },
  'S': { number: 16, name: 'Sulfur', group: 'nonmetal' },
  'Cl': { number: 17, name: 'Chlorine', group: 'halogen' },
  'Ar': { number: 18, name: 'Argon', group: 'noble-gas' }
};

const PeriodicTable = ({ elements }) => {
  const getElementStatus = (symbol) => {
    if (!elements[symbol]) return 'unemerged';
    const stability = elements[symbol].stability;
    if (stability >= 0.9) return 'stable';
    if (stability >= 0.7) return 'forming';
    return 'unstable';
  };

  const getElementOpacity = (symbol) => {
    if (!elements[symbol]) return 0.2;
    return Math.min(elements[symbol].count / 1000, 1.0);
  };

  return (
    <div className="periodic-table-container">
      <div className="periodic-table">
        {Object.entries(ELEMENT_INFO).map(([symbol, info]) => {
          const status = getElementStatus(symbol);
          const opacity = getElementOpacity(symbol);
          const elementData = elements[symbol];

          return (
            <div
              key={symbol}
              className={`element ${info.group} ${status}`}
              style={{ 
                gridColumn: info.number <= 2 ? info.number : (info.number <= 10 ? info.number + 10 : info.number - 8),
                gridRow: info.number <= 2 ? 1 : (info.number <= 10 ? 2 : 3),
                opacity: 0.3 + (opacity * 0.7)
              }}
              title={`${info.name} (${info.number})\n${elementData ? `Count: ${elementData.count}\nStability: ${(elementData.stability * 100).toFixed(1)}%` : 'Not emerged'}`}
            >
              <div className="element-number">{info.number}</div>
              <div className="element-symbol">{symbol}</div>
              <div className="element-name">{info.name}</div>
              {elementData && (
                <div className="element-count">{elementData.count}</div>
              )}
            </div>
          );
        })}
      </div>

      <div className="table-legend">
        <div className="legend-item">
          <span className="legend-dot stable"></span>
          <span>Stable (≥90%)</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot forming"></span>
          <span>Forming (70-90%)</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot unstable"></span>
          <span>Unstable (&lt;70%)</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot unemerged"></span>
          <span>Not Emerged</span>
        </div>
      </div>

      <div className="emergence-stats">
        <div className="stat">
          <span className="stat-label">Elements Emerged:</span>
          <span className="stat-value">{Object.keys(elements).length} / 18</span>
        </div>
        <div className="stat">
          <span className="stat-label">Total Particles:</span>
          <span className="stat-value">
            {Object.values(elements).reduce((sum, el) => sum + el.count, 0).toLocaleString()}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Avg Stability:</span>
          <span className="stat-value">
            {Object.keys(elements).length > 0
              ? (Object.values(elements).reduce((sum, el) => sum + el.stability, 0) / Object.keys(elements).length * 100).toFixed(1)
              : 0}%
          </span>
        </div>
      </div>
    </div>
  );
};

export default PeriodicTable;
