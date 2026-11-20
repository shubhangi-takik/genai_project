import React, { createContext, useContext, useMemo, useState } from 'react';

const AutoencoderResultContext = createContext(null);

export const AutoencoderResultProvider = ({ children }) => {
  const [result, setResult] = useState(null);
  const value = useMemo(() => ({ result, setResult }), [result]);
  return (
    <AutoencoderResultContext.Provider value={value}>
      {children}
    </AutoencoderResultContext.Provider>
  );
};

export const useAutoencoderResult = () => {
  const context = useContext(AutoencoderResultContext);
  if (!context) {
    throw new Error('useAutoencoderResult must be used within AutoencoderResultProvider');
  }
  return context;
};
