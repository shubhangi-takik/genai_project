import React from 'react';
import { ImageIcon } from 'lucide-react';

const ImageComparison = ({
  inputImage,
  outputImage,
  inputLabel = 'Input',
  outputLabel = 'Output',
  emptyOutputMessage = 'Model output will appear here after backend integration',
}) => {
  return (
    <div className="image-comparison-container">
      <div className="image-card">
        <h4>{inputLabel}</h4>
        <div className="image-wrapper">
          {inputImage ? (
            <img src={inputImage} alt={inputLabel} />
          ) : (
            <div className="empty-image">
              <ImageIcon size={48} className="placeholder-icon" />
              <span>Waiting for input...</span>
            </div>
          )}
        </div>
      </div>
      
      <div className="image-card">
        <h4>{outputLabel}</h4>
        <div className="image-wrapper">
          {outputImage ? (
            <img src={outputImage} alt={outputLabel} />
          ) : (
            <div className="empty-image">
              <ImageIcon size={48} className="placeholder-icon" />
              <span>{emptyOutputMessage}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageComparison;
