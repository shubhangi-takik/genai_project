import React from 'react';
import { Loader } from 'lucide-react';

const LoadingState = ({ message = 'Processing...' }) => {
  return (
    <div className="loading-state">
      <Loader className="spinner" size={32} />
      <p>{message}</p>
    </div>
  );
};

export default LoadingState;
