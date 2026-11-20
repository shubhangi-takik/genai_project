import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const ChartCard = ({ title, data, lines = [] }) => {
  return (
    <div className="chart-card">
      <h3 className="chart-title">{title}</h3>
      <div className="chart-container">
        {!data || data.length === 0 ? (
          <div className="no-data-placeholder">
            <p>No results available</p>
            <span>Run experiment to generate data</span>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc' }} 
              />
              <Legend />
              {lines.map((line, index) => (
                <Line 
                  key={index} 
                  type="monotone" 
                  dataKey={line.dataKey} 
                  stroke={line.color} 
                  activeDot={{ r: 8 }} 
                  strokeWidth={2}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};

export default ChartCard;
