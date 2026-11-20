import React from 'react';

const MetricCard = ({ title, value, unit = '' }) => {
  return (
    <div className="metric-card">
      <h3 className="metric-title">{title}</h3>
      <div className="metric-value">
        {value === null || value === undefined ? (
          <span className="no-data">N/A</span>
        ) : (
          <>
            {value} <span className="metric-unit">{unit}</span>
          </>
        )}
      </div>
    </div>
  );
};

export default MetricCard;
