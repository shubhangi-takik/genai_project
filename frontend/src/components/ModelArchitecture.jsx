import React from 'react';

const ModelArchitecture = ({ title, description, flowDiagram }) => {
  return (
    <div className="architecture-card">
      <h3>{title}</h3>
      <p>{description}</p>
      <div className="flow-diagram">
        {flowDiagram.map((step, index) => (
          <React.Fragment key={index}>
            <div className="flow-step">{step}</div>
            {index < flowDiagram.length - 1 && <div className="flow-arrow">→</div>}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default ModelArchitecture;
