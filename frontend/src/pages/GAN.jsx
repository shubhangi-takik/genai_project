import React, { useEffect, useState } from 'react';
import ModelArchitecture from '../components/ModelArchitecture';
import ImageComparison from '../components/ImageComparison';
import ImageUpload from '../components/ImageUpload';
import MetricCard from '../components/MetricCard';
import ChartCard from '../components/ChartCard';
import LoadingState from '../components/LoadingState';
import { generateGan, getGanHistory, getGanStatus } from '../services/api';

const formatMetric = (value, digits = 4) => {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return null;
  return Number(value).toFixed(digits);
};

const GAN = () => {
  const [image, setImage] = useState(null);
  const [inputPreviewUrl, setInputPreviewUrl] = useState('');
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [isWorking, setIsWorking] = useState(false);
  const [error, setError] = useState('');
  const [statusError, setStatusError] = useState('');

  useEffect(() => {
    if (!inputPreviewUrl) return undefined;
    return () => URL.revokeObjectURL(inputPreviewUrl);
  }, [inputPreviewUrl]);

  useEffect(() => {
    let active = true;
    Promise.all([getGanStatus(), getGanHistory()])
      .then(([modelStatus, trainingHistory]) => {
        if (!active) return;
        setStatus(modelStatus);
        setHistory(trainingHistory);
      })
      .catch((requestError) => {
        if (active) setStatusError(requestError.message || 'Could not load GAN status.');
      });
    return () => {
      active = false;
    };
  }, []);

  const handleImageUploaded = (file) => {
    setImage(file);
    setInputPreviewUrl(file ? URL.createObjectURL(file) : '');
    setResult(null);
    setError('');
  };

  const runGeneration = async () => {
    if (!image) {
      setError('Upload an interior image to run GAN inference.');
      return;
    }
    setError('');
    setIsWorking(true);
    const formData = new FormData();
    formData.append('image', image);
    try {
      setResult(await generateGan(formData));
    } catch (requestError) {
      setError(requestError.message || 'GAN generation failed.');
    } finally {
      setIsWorking(false);
    }
  };

  const metrics = result?.metrics;
  const modelAvailable = status?.available === true;

  return (
    <div className="page-container">
      <h2 className="page-title">Generative Adversarial Network (GAN)</h2>
      <p className="page-description">
        Run the trained conditional GAN on the backend to generate an image from the uploaded room&apos;s
        structural edges.
      </p>

      <ModelArchitecture
        title="Model Architecture: Conditional Pix2Pix (U-Net + PatchGAN)"
        description="The U-Net generator reconstructs room appearance from an edge-map condition. This checkpoint operates at 128 x 128 pixels and was not trained to follow text prompts or add specific furniture."
        flowDiagram={[
          'Input Room Image',
          'Structural Edge Map',
          'U-Net Generator',
          'Generated Room Image',
        ]}
      />

      <div className="card" style={{ marginBottom: '2rem' }}>
        <h3 className="section-title" style={{ marginTop: 0 }}>Generate from a room image</h3>
        {statusError ? (
          <p className="page-description" role="alert" style={{ color: '#f87171' }}>
            {statusError}
          </p>
        ) : (
          <p className="page-description" role="status">
            {modelAvailable
              ? `GAN checkpoint ready on ${status.device}.`
              : 'GAN checkpoint is unavailable. Start the backend with the trained best_gan.pt checkpoint.'}
          </p>
        )}
        <ImageUpload onImageUpload={handleImageUploaded} />
        <button
          type="button"
          className="btn-primary"
          onClick={runGeneration}
          disabled={!image || !modelAvailable || isWorking}
          style={{ marginTop: '1rem' }}
        >
          {isWorking ? 'Generating...' : 'Generate with GAN'}
        </button>
        {isWorking && <LoadingState message="Running the trained GAN on the backend..." />}
        {error && (
          <p className="page-description" role="alert" style={{ color: '#f87171', marginTop: '1rem' }}>
            {error}
          </p>
        )}
      </div>

      <h3 className="section-title">Results: Input Room vs. GAN Output</h3>
      <ImageComparison
        inputImage={inputPreviewUrl || result?.input_image || null}
        outputImage={result?.output_image || null}
        inputLabel="Input Room"
        outputLabel="GAN Output"
        emptyOutputMessage="A generated image will appear here after GAN inference completes."
      />

      {result?.output_image && (
        <a
          className="btn-primary"
          href={result.output_image}
          download="gan-generated.png"
          style={{ display: 'inline-flex', marginTop: '0.75rem', textDecoration: 'none' }}
        >
          Download GAN Output
        </a>
      )}

      {result?.condition_image && (
        <div className="image-card" style={{ marginTop: '1rem' }}>
          <h4>Structural Condition Used by the Model</h4>
          <div className="image-wrapper">
            <img src={result.condition_image} alt="Structural edge-map condition" />
          </div>
        </div>
      )}

      {result && (
        <>
          <h3 className="section-title">Inference Metrics</h3>
          <div className="grid-4" style={{ marginBottom: '2rem' }}>
            <MetricCard title="SSIM vs. Input" value={formatMetric(metrics?.ssim)} />
            <MetricCard title="PSNR vs. Input" value={formatMetric(metrics?.psnr, 2)} unit="dB" />
            <MetricCard
              title="Generator Loss"
              value={formatMetric(metrics?.generator_loss)}
            />
            <MetricCard
              title="Discriminator Loss"
              value={formatMetric(metrics?.discriminator_loss)}
            />
            <MetricCard
              title="Inference Time"
              value={formatMetric(metrics?.inference_time_ms, 2)}
              unit="ms"
            />
          </div>
        </>
      )}

      <h3 className="section-title">Training History</h3>
      <div className="grid-2">
        <ChartCard
          title="Generator Loss"
          data={history}
          lines={[
            { dataKey: 'generator_loss', color: '#3b82f6' },
            { dataKey: 'val_g_loss', color: '#8b5cf6' },
          ]}
        />
        <ChartCard
          title="Discriminator Loss"
          data={history}
          lines={[
            { dataKey: 'discriminator_loss', color: '#ef4444' },
            { dataKey: 'val_d_loss', color: '#f97316' },
          ]}
        />
      </div>
    </div>
  );
};

export default GAN;
