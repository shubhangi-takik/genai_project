import React, { useEffect, useState } from 'react';
import ModelArchitecture from '../components/ModelArchitecture';
import ImageComparison from '../components/ImageComparison';
import MetricCard from '../components/MetricCard';
import ChartCard from '../components/ChartCard';
import { useAutoencoderResult } from '../context/AutoencoderResultContext';
import { getAutoencoderHistory } from '../services/api';

const formatMetric = (value, digits) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return null;
  return Number(value).toFixed(digits);
};

const Autoencoder = () => {
  const { result } = useAutoencoderResult();
  const [history, setHistory] = useState([]);
  const metrics = result?.metrics;

  useEffect(() => {
    getAutoencoderHistory().then(setHistory);
  }, []);

  return (
    <div className="page-container">
      <h2 className="page-title">Autoencoder</h2>
      <p className="page-description">
        Compresses room images into a latent space and reconstructs them to learn fundamental structural features.
      </p>

      <ModelArchitecture 
        title="Architecture"
        description="The Autoencoder consists of an encoder network that maps the input image to a lower-dimensional representation, and a decoder network that reconstructs the image from this latent representation."
        flowDiagram={['Input Image', 'Encoder', 'Latent Representation', 'Decoder', 'Reconstructed Image']}
      />

      <h3 className="section-title">Results</h3>
      <ImageComparison 
        inputLabel="Input Image"
        outputLabel="Reconstructed Image"
        inputImage={result?.input_image || null}
        outputImage={result?.output_image || null}
      />

      <h3 className="section-title">Performance Metrics</h3>
      <div className="grid-4" style={{ marginBottom: '2rem' }}>
        <MetricCard title="MSE" value={formatMetric(metrics?.mse, 6)} />
        <MetricCard title="MAE" value={formatMetric(metrics?.mae, 6)} />
        <MetricCard title="PSNR" value={formatMetric(metrics?.psnr, 2)} unit="dB" />
        <MetricCard title="SSIM" value={formatMetric(metrics?.ssim, 4)} />
        <MetricCard title="Reconstruction Loss" value={formatMetric(metrics?.reconstruction_loss, 6)} />
        <MetricCard title="Inference Time" value={formatMetric(metrics?.inference_time_ms, 2)} unit="ms" />
      </div>

      <h3 className="section-title">Training History</h3>
      <div className="grid-2">
        <ChartCard 
          title="Training vs Validation Loss" 
          data={history}
          lines={[
            { dataKey: 'train', color: '#3b82f6' },
            { dataKey: 'val', color: '#8b5cf6' },
          ]}
        />
      </div>
    </div>
  );
};

export default Autoencoder;
