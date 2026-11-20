import React, { useState } from 'react';
import ModelArchitecture from '../components/ModelArchitecture';
import ImageComparison from '../components/ImageComparison';
import ImageUpload from '../components/ImageUpload';
import LoadingState from '../components/LoadingState';
import { generateDiffusionImage } from '../services/api';

const Diffusion = () => {
  const [image, setImage] = useState(null);
  const [roomType, setRoomType] = useState('living room');
  const [interiorStyle, setInteriorStyle] = useState('modern');
  const [colorPreference, setColorPreference] = useState('warm beige');
  const [furniturePreference, setFurniturePreference] = useState('sofa, coffee table, rug');
  const [designPrompt, setDesignPrompt] = useState(
    'Transform this empty room into a photorealistic modern living room with a beige sofa, wood coffee table, rug, curtains, indoor plants, and warm ambient lighting while preserving the original walls, windows, flooring, and perspective.'
  );
  const [result, setResult] = useState(null);
  const [isWorking, setIsWorking] = useState(false);
  const [error, setError] = useState('');

  const generate = async () => {
    setError('');
    if (!image) {
      setError('Upload an empty-room image before running diffusion inpainting.');
      return;
    }
    if (!designPrompt.trim()) {
      setError('Enter a design prompt before generating the furnished room.');
      return;
    }

    const formData = new FormData();
    formData.append('image', image);
    formData.append('prompt', designPrompt.trim());
    formData.append('room_type', roomType.trim());
    formData.append('interior_style', interiorStyle.trim());
    formData.append('color_preference', colorPreference.trim());
    formData.append('furniture_preference', furniturePreference.trim());

    setIsWorking(true);
    try {
      const response = await generateDiffusionImage(formData);
      setResult(response);
    } catch (requestError) {
      setError(requestError.message || 'Diffusion generation failed.');
    } finally {
      setIsWorking(false);
    }
  };

  return (
    <div className="page-container">
      <h2 className="page-title">Diffusion Model</h2>
      <p className="page-description">
        Upload an empty room and produce a furnished version of the same room with a local image-conditioned inpainting pipeline.
      </p>

      <ModelArchitecture
        title="Architecture"
        description="A pretrained image-to-image inpainting model conditions on the original room image and prompt, then fills the selected region with furniture and décor while preserving the room geometry as much as possible."
        flowDiagram={['Uploaded Room Image', 'Inpainting Mask', 'Text Prompt + Style', 'Diffusion Denoising', 'Furnished Room Output']}
      />

      <div className="card" style={{ marginBottom: '2rem' }}>
        <h3 className="section-title" style={{ marginTop: 0 }}>Room setup</h3>
        <ImageUpload onImageUpload={setImage} />

        <div className="grid-2" style={{ marginTop: '1.5rem' }}>
          <div className="form-group">
            <label htmlFor="diffusion-room-type">Room type</label>
            <select
              id="diffusion-room-type"
              className="form-control"
              value={roomType}
              onChange={(event) => setRoomType(event.target.value)}
            >
              <option value="living room">Living room</option>
              <option value="bedroom">Bedroom</option>
              <option value="kitchen">Kitchen</option>
              <option value="bathroom">Bathroom</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="diffusion-style">Interior style</label>
            <select
              id="diffusion-style"
              className="form-control"
              value={interiorStyle}
              onChange={(event) => setInteriorStyle(event.target.value)}
            >
              <option value="modern">Modern</option>
              <option value="scandinavian">Scandinavian</option>
              <option value="industrial">Industrial</option>
              <option value="boho">Boho</option>
              <option value="minimalist">Minimalist</option>
            </select>
          </div>
        </div>

        <div className="grid-2">
          <div className="form-group">
            <label htmlFor="diffusion-colors">Colour preference</label>
            <input
              id="diffusion-colors"
              className="form-control"
              value={colorPreference}
              onChange={(event) => setColorPreference(event.target.value)}
              placeholder="Warm beige, walnut wood, white and cream"
            />
          </div>

          <div className="form-group">
            <label htmlFor="diffusion-furniture">Furniture preference</label>
            <input
              id="diffusion-furniture"
              className="form-control"
              value={furniturePreference}
              onChange={(event) => setFurniturePreference(event.target.value)}
              placeholder="Sofa, lounge chair, coffee table, rug"
            />
          </div>
        </div>

        <div className="form-group">
          <label htmlFor="diffusion-prompt">Design prompt</label>
          <textarea
            id="diffusion-prompt"
            className="form-control"
            rows="5"
            value={designPrompt}
            onChange={(event) => setDesignPrompt(event.target.value)}
            placeholder="Describe the desired interior design, furniture, colours, and lighting."
          />
        </div>

        <div className="grid-2" style={{ marginTop: '1rem' }}>
          <button type="button" className="btn-primary" onClick={generate} disabled={isWorking}>
            {isWorking ? 'Generating...' : 'Generate Furnished Room'}
          </button>
          {result?.output_image && (
            <a className="btn-primary" href={result.output_image} target="_blank" rel="noreferrer" style={{ textAlign: 'center', textDecoration: 'none' }}>
              Download Result
            </a>
          )}
        </div>

        {isWorking && <LoadingState message="Running local diffusion inpainting on the uploaded room image..." />}
        {error && (
          <p className="page-description" role="alert" style={{ color: '#f87171', marginTop: '1rem' }}>
            {error}
          </p>
        )}
      </div>

      <h3 className="section-title">Results</h3>
      <ImageComparison
        inputLabel="Original Room"
        outputLabel="Generated Furnished Room"
        inputImage={image ? URL.createObjectURL(image) : null}
        outputImage={result?.output_image || null}
        emptyOutputMessage="The generated furnished room will appear here after a successful diffusion run."
      />

      {result?.prompt && (
        <div className="card" style={{ marginTop: '1.5rem' }}>
          <h4 className="section-title" style={{ marginTop: 0 }}>Prompt used for generation</h4>
          <p className="page-description" style={{ whiteSpace: 'pre-wrap' }}>{result.prompt}</p>
        </div>
      )}
    </div>
  );
};

export default Diffusion;
