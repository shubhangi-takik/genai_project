import React from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const ComparisonChartCard = ({ title, data, unit = '', lowerIsBetter = false }) => (
  <div className="chart-card">
    <h3 className="chart-title">{title}</h3>
    <p className="page-description" style={{ marginBottom: '0.5rem' }}>
      {lowerIsBetter ? 'Lower is better' : 'Higher is better'}
    </p>
    <div className="chart-container">
      {!data?.length ? (
        <div className="no-data-placeholder">
          <p>No comparable measurements</p>
          <span>Only measured values are charted.</span>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="name" stroke="#94a3b8" />
            <YAxis
              stroke="#94a3b8"
              tickFormatter={(value) => `${value}${unit}`}
            />
            <Tooltip
              formatter={(value) => [`${Number(value).toPrecision(5)}${unit}`, title]}
              contentStyle={{
                backgroundColor: '#1e293b',
                border: 'none',
                borderRadius: '8px',
                color: '#f8fafc',
              }}
            />
            <Bar dataKey="value" fill="#8b5cf6" radius={[5, 5, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  </div>
);

export default ComparisonChartCard;
