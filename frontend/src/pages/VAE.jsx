import React, { useEffect, useState } from 'react';
import ModelArchitecture from '../components/ModelArchitecture';
import ImageComparison from '../components/ImageComparison';
import ImageUpload from '../components/ImageUpload';
import MetricCard from '../components/MetricCard';
import ChartCard from '../components/ChartCard';
import LoadingState from '../components/LoadingState';
import {
  generateVaeSample,
  getVaeHistory,
  getVaeStatus,
  reconstructVae,
} from '../services/api';

const formatMetric = (value, digits = 4) => {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return null;
  return Number(value).toFixed(digits);
};

const VAE = () => {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [status, setStatus] = useState(null);
  const [seed, setSeed] = useState('');
  const [isWorking, setIsWorking] = useState(false);
  const [error, setError] = useState('');
  const [statusError, setStatusError] = useState('');

  useEffect(() => {
    let active = true;
    Promise.all([getVaeStatus(), getVaeHistory()])
      .then(([modelStatus, trainingHistory]) => {
        if (!active) return;
        setStatus(modelStatus);
        setHistory(trainingHistory);
      })
      .catch((requestError) => {
        if (active) setStatusError(requestError.message || 'Could not load VAE status.');
      });
    return () => {
      active = false;
    };
  }, []);

  const runReconstruction = async () => {
    setError('');
    if (!image) {
      setError('Upload an interior image to reconstruct it.');
      return;
    }
    const formData = new FormData();
    formData.append('image', image);
    setIsWorking(true);
    try {
      setResult(await reconstructVae(formData));
    } catch (requestError) {
      setError(requestError.message || 'VAE reconstruction failed.');
    } finally {
      setIsWorking(false);
    }
  };

  const runGeneration = async () => {
    setError('');
    setIsWorking(true);
    try {
      setResult(await generateVaeSample(seed === '' ? null : Number(seed)));
    } catch (requestError) {
      setError(requestError.message || 'VAE sampling failed.');
    } finally {
      setIsWorking(false);
    }
  };

  const metrics = result?.metrics;
  const modelAvailable = status?.available === true;

  return (
    <div className="page-container">
      <h2 className="page-title">Variational Autoencoder (VAE)</h2>
      <p className="page-description">
        Reconstruct uploaded interiors or sample new images from the VAE&apos;s learned latent prior.
        Sampling is unconditional; prompts and room/style labels do not control the output.
      </p>

      <ModelArchitecture
        title="Architecture"
        description="A convolutional encoder predicts latent mean and log-variance. The reparameterization trick enables training, while a convolutional decoder reconstructs or samples images."
        flowDiagram={['Input Image', 'Conv Encoder', 'μ and log σ²', 'Latent z', 'Conv Decoder', 'Interior Image']}
      />

      <div className="card">
        <h3 className="section-title" style={{ marginTop: 0 }}>Generate or reconstruct</h3>
        {statusError && (
          <p className="page-description" role="alert" style={{ color: '#f87171' }}>
            {statusError}
          </p>
        )}
        {!statusError && (
          <p className="page-description" role="status" style={{ marginBottom: '1rem' }}>
            {modelAvailable
              ? `VAE checkpoint ready on ${status.device}.`
              : 'VAE checkpoint is not available yet. Train the VAE before generating or reconstructing images.'}
          </p>
        )}

        <ImageUpload onImageUpload={setImage} />
        <div className="form-group">
          <label htmlFor="vae-seed">Sampling seed (optional, for repeatable output)</label>
          <input
            id="vae-seed"
            className="form-control"
            type="number"
            min="0"
            max="2147483647"
            value={seed}
            onChange={(event) => setSeed(event.target.value)}
            placeholder="Leave blank for a random seed"
          />
        </div>
        <div className="grid-2">
          <button
            type="button"
            className="btn-primary"
            onClick={runGeneration}
            disabled={!modelAvailable || isWorking}
          >
            {isWorking ? 'Generating...' : 'Generate New VAE Sample'}
          </button>
          <button
            type="button"
            className="btn-primary"
            onClick={runReconstruction}
            disabled={!modelAvailable || !image || isWorking}
          >
            {isWorking ? 'Reconstructing...' : 'Reconstruct Uploaded Image'}
          </button>
        </div>
        {isWorking && <LoadingState message="Running VAE inference on the backend..." />}
        {error && (
          <p className="page-description" role="alert" style={{ color: '#f87171', marginTop: '1rem' }}>
            {error}
          </p>
        )}
      </div>

      <h3 className="section-title">Results</h3>
      <ImageComparison
        inputLabel={result?.mode === 'generation' ? 'Prior sample' : 'Input Image'}
        outputLabel={result?.mode === 'generation' ? 'New VAE Sample' : 'VAE Reconstruction'}
        inputImage={result?.input_image || null}
        outputImage={result?.output_image || null}
        emptyOutputMessage="A VAE result will appear here after a successful request."
      />
      {result?.mode === 'generation' && result.seed !== null && (
        <p className="page-description">Generated with latent sampling seed {result.seed}.</p>
      )}

      {result?.mode === 'reconstruction' && (
        <>
          <h3 className="section-title">Reconstruction Metrics</h3>
          <div className="grid-4" style={{ marginBottom: '2rem' }}>
            <MetricCard title="MSE" value={formatMetric(metrics?.mse, 6)} />
            <MetricCard title="MAE" value={formatMetric(metrics?.mae, 6)} />
            <MetricCard title="PSNR" value={formatMetric(metrics?.psnr, 2)} unit="dB" />
            <MetricCard title="SSIM" value={formatMetric(metrics?.ssim, 4)} />
            <MetricCard title="KL Divergence" value={formatMetric(metrics?.kl_divergence, 4)} />
            <MetricCard title="Total VAE Loss" value={formatMetric(metrics?.total_loss, 4)} />
            <MetricCard title="Inference Time" value={formatMetric(metrics?.inference_time_ms, 2)} unit="ms" />
          </div>
        </>
      )}

      <h3 className="section-title">Training History</h3>
      <div className="grid-2">
        <ChartCard
          title="Total VAE Loss"
          data={history}
          lines={[
            { dataKey: 'train_total', color: '#3b82f6' },
            { dataKey: 'val_total', color: '#8b5cf6' },
          ]}
        />
        <ChartCard
          title="Reconstruction Loss"
          data={history}
          lines={[
            { dataKey: 'train_reconstruction', color: '#3b82f6' },
            { dataKey: 'val_reconstruction', color: '#8b5cf6' },
          ]}
        />
        <ChartCard
          title="KL Divergence"
          data={history}
          lines={[
            { dataKey: 'train_kl', color: '#3b82f6' },
            { dataKey: 'val_kl', color: '#8b5cf6' },
          ]}
        />
      </div>
    </div>
  );
};

export default VAE;
