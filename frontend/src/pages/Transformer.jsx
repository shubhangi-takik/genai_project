import React, { useState } from 'react';
import PromptInput from '../components/PromptInput';
import { analyzeDesignPrompt } from '../services/api';

const REQUIREMENT_FIELDS = [
  ['room_type', 'Room Type'],
  ['interior_style', 'Interior Style'],
  ['color_preferences', 'Colour Preferences'],
  ['furniture_preferences', 'Furniture Preferences'],
  ['lighting_preferences', 'Lighting Preferences'],
  ['design_keywords', 'Design Keywords'],
];

const displayValue = (value) => {
  if (Array.isArray(value)) return value.length ? value.join(', ') : 'No matching terms found';
  return value || 'Not identified';
};

const Transformer = () => {
  const [prompt, setPrompt] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [requirements, setRequirements] = useState(null);
  const [error, setError] = useState('');

  const handleAnalyze = async () => {
    const trimmedPrompt = prompt.trim();
    setError('');
    if (!trimmedPrompt) {
      setRequirements(null);
      setError('Enter a design prompt before analyzing requirements.');
      return;
    }

    setIsAnalyzing(true);
    try {
      const result = await analyzeDesignPrompt(trimmedPrompt);
      setRequirements(result);
    } catch (requestError) {
      setRequirements(null);
      setError(requestError.message || 'Could not analyze the design prompt.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="page-container">
      <h2 className="page-title">Transformer (Text Understanding)</h2>
      <p className="page-description">
        Extracts structured interior-design requirements from a natural-language design prompt.
        This analysis does not generate an image.
      </p>

      <div className="card" style={{ marginBottom: '2rem' }}>
        <h3 className="section-title" style={{ marginTop: 0 }}>Natural Language Input</h3>
        <PromptInput
          value={prompt}
          onChange={(event) => {
            setPrompt(event.target.value);
            setError('');
          }}
          placeholder="e.g., Design a modern living room with a beige sofa, wooden coffee table, plants, and warm lighting."
        />
        <button
          type="button"
          className="btn-primary"
          onClick={handleAnalyze}
          disabled={isAnalyzing}
          style={{ marginTop: '1rem' }}
        >
          {isAnalyzing ? 'Analyzing...' : 'Analyze Requirements'}
        </button>
        {error && (
          <p className="page-description" role="alert" style={{ color: '#f87171', marginTop: '1rem' }}>
            {error}
          </p>
        )}
      </div>

      <h3 className="section-title">Structured Requirements</h3>
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div className="grid-2">
          {REQUIREMENT_FIELDS.map(([key, label]) => (
            <div key={key}>
              <strong>{label}:</strong>{' '}
              <span className={requirements ? '' : 'no-data'}>
                {requirements
                  ? displayValue(requirements.requirements[key])
                  : 'Waiting for analysis...'}
              </span>
            </div>
          ))}
        </div>
        {requirements && (
          <p className="page-description" role="status" style={{ marginTop: '1rem' }}>
            Extraction method: {requirements.extraction_method}. This is a deterministic
            rule-based baseline, not a trained Transformer NLP model.
          </p>
        )}
      </div>
    </div>
  );
};

export default Transformer;
