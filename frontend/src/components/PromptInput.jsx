import React from 'react';

const PromptInput = ({ value, onChange, placeholder = "Enter your design prompt..." }) => {
  return (
    <div className="prompt-input-container">
      <label htmlFor="design-prompt">Design Prompt</label>
      <textarea
        id="design-prompt"
        className="prompt-textarea"
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        rows={4}
      />
    </div>
  );
};

export default PromptInput;
