import React, { useState } from 'react';
import { UploadCloud } from 'lucide-react';

const ImageUpload = ({ onImageUpload }) => {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    if (onImageUpload) {
      onImageUpload(file);
    }
  };

  return (
    <div className="image-upload-container">
      <div 
        className={`upload-zone ${dragActive ? 'active' : ''} ${preview ? 'has-image' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input type="file" id="file-upload" multiple={false} accept="image/*" onChange={handleChange} />
        <label htmlFor="file-upload" className={preview ? 'hidden' : ''}>
          <UploadCloud size={48} className="upload-icon" />
          <p>Drag and drop your room image here</p>
          <span>or click to browse</span>
        </label>
        
        {preview && (
          <div className="preview-container">
            <img src={preview} alt="Upload preview" className="image-preview" />
            <button
              type="button"
              className="change-image-btn"
              onClick={() => {
                setPreview(null);
                if (onImageUpload) onImageUpload(null);
              }}
            >
              Change Image
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUpload;
