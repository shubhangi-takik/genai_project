import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ImageIcon } from 'lucide-react';
import ImageUpload from '../components/ImageUpload';
import PromptInput from '../components/PromptInput';
import LoadingState from '../components/LoadingState';
import { generateDesign, generateInteriorImage } from '../services/api';
import { useAutoencoderResult } from '../context/AutoencoderResultContext';

const Home = () => {
  const [image, setImage] = useState(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState('');
  const [roomType, setRoomType] = useState('bedroom');
  const [interiorStyle, setInteriorStyle] = useState('modern');
  const [colorPref, setColorPref] = useState('');
  const [furniturePref, setFurniturePref] = useState('');
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState('');
  const [generatedImage, setGeneratedImage] = useState(null);
  const navigate = useNavigate();
  const { setResult } = useAutoencoderResult();

  useEffect(() => {
    if (!image) {
      setUploadedImageUrl('');
      return undefined;
    }
    const imageUrl = URL.createObjectURL(image);
    setUploadedImageUrl(imageUrl);
    return () => URL.revokeObjectURL(imageUrl);
  }, [image]);

  const handleGenerate = async () => {
    setError('');
    if (!prompt.trim()) {
      setError('Enter a design prompt before generating an image.');
      return;
    }

    setIsGenerating(true);
    setGeneratedImage(null);
    try {
      const payload = await generateInteriorImage({
        prompt,
        room_type: roomType,
        interior_style: interiorStyle,
        color_preference: colorPref,
        furniture_preference: furniturePref,
      });
      setGeneratedImage(payload);
    } catch (err) {
      setError(err.message || 'Image generation failed.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleReconstruction = async () => {
    setError('');
    if (!image) {
      setError('Upload a room image before running the Autoencoder reconstruction.');
      return;
    }

    const formData = new FormData();
    formData.append('image', image);
    formData.append('room_type', roomType);
    formData.append('interior_style', interiorStyle);
    formData.append('color_preference', colorPref);
    formData.append('furniture_preference', furniturePref);
    formData.append('design_prompt', prompt);

    setIsGenerating(true);
    try {
      const payload = await generateDesign(formData);
      setResult(payload);
      navigate('/autoencoder');
    } catch (err) {
      setError(err.message || 'Autoencoder inference failed.');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="page-container">
      <h2 className="page-title">Multi-Model Generative AI-Based Virtual Interior Design System</h2>
      <p className="page-description">
        An AI-powered system for generating personalized virtual interior designs using multiple Generative AI models.
      </p>

      <div className="card">
        <ImageUpload onImageUpload={setImage} />
        
        <div className="grid-2">
          <div className="form-group">
            <label htmlFor="room-type">Room Type</label>
            <select 
              id="room-type" 
              className="form-control"
              value={roomType}
              onChange={(e) => setRoomType(e.target.value)}
            >
              <option value="bedroom">Bedroom</option>
              <option value="living">Living Room</option>
              <option value="kitchen">Kitchen</option>
              <option value="bathroom">Bathroom</option>
              <option value="office">Office</option>
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="interior-style">Interior Style</label>
            <select 
              id="interior-style" 
              className="form-control"
              value={interiorStyle}
              onChange={(e) => setInteriorStyle(e.target.value)}
            >
              <option value="modern">Modern</option>
              <option value="minimalist">Minimalist</option>
              <option value="industrial">Industrial</option>
              <option value="scandinavian">Scandinavian</option>
              <option value="classic">Classic</option>
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="color-pref">Color Preference</label>
            <input 
              type="text" 
              id="color-pref" 
              className="form-control" 
              placeholder="e.g., Warm tones, blues"
              value={colorPref}
              onChange={(e) => setColorPref(e.target.value)}
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="furniture-pref">Furniture Preference</label>
            <input 
              type="text" 
              id="furniture-pref" 
              className="form-control" 
              placeholder="e.g., Wooden bed, velvet sofa"
              value={furniturePref}
              onChange={(e) => setFurniturePref(e.target.value)}
            />
          </div>
        </div>

        <PromptInput 
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Modern bedroom with wooden furniture, warm lighting and minimal decoration."
        />

        {error && (
          <p className="page-description" style={{ color: '#f87171', marginTop: '1rem' }}>
            {error}
          </p>
        )}

        <p className="page-description" style={{ marginTop: '1rem' }}>
          Generates a new image from your prompt using the hosted OpenAI image API. Each generation is a paid API request; cost varies with image output.
        </p>
        <button 
          className="btn-primary" 
          onClick={handleGenerate}
          disabled={isGenerating}
          style={{ marginTop: '1rem' }}
        >
          {isGenerating ? 'Generating image...' : 'Generate Design'}
        </button>
        {isGenerating && <LoadingState message="Generating a new interior image from your prompt..." />}
        {image && (
          <button
            type="button"
            className="btn-primary"
            onClick={handleReconstruction}
            disabled={isGenerating}
            style={{ marginTop: '1rem', marginLeft: '0.75rem' }}
          >
            Reconstruct Uploaded Image (Autoencoder)
          </button>
        )}
      </div>

      <section className="design-model-overview" aria-label="Reconstruction and image generation">
        <article className="design-model-row">
          <div className="design-model-thumbnail">
            {uploadedImageUrl ? (
              <img src={uploadedImageUrl} alt="Uploaded room for Autoencoder reconstruction" />
            ) : (
              <div className="design-model-empty">
                <ImageIcon size={36} />
                <span>Upload a room image</span>
              </div>
            )}
          </div>
          <div className="design-model-copy">
            <h3>Current: Autoencoder</h3>
            <span className="design-model-badge reconstruction">Reconstruction</span>
            <p>
              Recreates an existing uploaded image. It does not follow your text prompt or add furniture and decor.
            </p>
          </div>
        </article>

        <article className="design-model-row">
          <div className="design-model-thumbnail">
            {generatedImage?.output_image ? (
              <img src={generatedImage.output_image} alt="Newly generated interior design" />
            ) : (
              <div className="design-model-empty">
                <ImageIcon size={36} />
                <span>{isGenerating ? 'Creating your design...' : 'Generated design will appear here'}</span>
              </div>
            )}
          </div>
          <div className="design-model-copy">
            <h3>Target: Generative AI</h3>
            <span className="design-model-badge new-design">New design</span>
            <p>
              Uses your prompt and selected room details to generate a new interior image.
            </p>
          </div>
        </article>
      </section>

      {generatedImage && (
        <section className="card" style={{ marginTop: '2rem' }}>
          <h3 className="section-title" style={{ marginTop: 0 }}>Generated Interior Design</h3>
          <p className="page-description">Generated by {generatedImage.model}</p>
          <img
            src={generatedImage.output_image}
            alt={`Generated ${generatedImage.room_type || 'interior'} design`}
            style={{ display: 'block', maxWidth: '100%', maxHeight: '720px', objectFit: 'contain', borderRadius: '12px' }}
          />
          <a href={generatedImage.output_image} download style={{ display: 'inline-block', marginTop: '1rem' }}>
            Download generated image
          </a>
        </section>
      )}
    </div>
  );
};

export default Home;
