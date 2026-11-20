import React from 'react';

const ModelCard = ({ title, description, to }) => {
  return (
    <div className="model-card">
      <h3>{title}</h3>
      <p>{description}</p>
      {/* Could add a link button here if needed */}
    </div>
  );
};

export default ModelCard;
