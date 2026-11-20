import React, { useEffect, useState } from 'react';
import ComparisonChartCard from '../components/ComparisonChartCard';
import { getModelComparison } from '../services/api';

const METRIC_COLUMNS = [
  { key: 'mse', label: 'MSE', digits: 6 },
  { key: 'mae', label: 'MAE', digits: 6 },
  { key: 'psnr', label: 'PSNR (dB)', digits: 2 },
  { key: 'ssim', label: 'SSIM', digits: 4 },
  { key: 'inference_time_ms', label: 'Inference time (ms)', digits: 2 },
];

const formatMetric = (value, digits) => (
  Number.isFinite(value) ? value.toFixed(digits) : '—'
);

const Comparison = () => {
  const [comparison, setComparison] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    getModelComparison()
      .then((result) => {
        if (active) setComparison(result);
      })
      .catch((requestError) => {
        if (active) setError(requestError.message || 'Could not load model evaluation results.');
      })
      .finally(() => {
        if (active) setIsLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const models = comparison?.models || [];
  const chartData = (metric) => models
    .filter((model) => model.status === 'evaluated' && Number.isFinite(model.metrics?.[metric]))
    .map((model) => ({ name: model.name, value: model.metrics[metric] }));

  return (
    <div className="page-container">
      <h2 className="page-title">Model Comparison Dashboard</h2>
      <p className="page-description">
        Measured evaluation results are loaded from the project&apos;s saved test-split reports.
        Metrics are shown only when the model has a relevant, valid measurement.
      </p>

      {isLoading && (
        <p className="page-description" role="status">Loading saved model evaluation results...</p>
      )}
      {error && (
        <div className="card" role="alert" style={{ marginBottom: '1.5rem' }}>
          <p className="page-description" style={{ color: '#f87171', margin: 0 }}>{error}</p>
        </div>
      )}

      {!isLoading && !error && (
        <>
          <div className="card" style={{ marginBottom: '2rem' }}>
            <h3 className="section-title" style={{ marginTop: 0 }}>Measured model scores</h3>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Task</th>
                    <th>Evaluation</th>
                    <th>Images</th>
                    {METRIC_COLUMNS.map((metric) => <th key={metric.key}>{metric.label}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {models.map((model) => (
                    <tr key={model.name}>
                      <td><strong>{model.name}</strong></td>
                      <td>{model.task}</td>
                      <td>{model.status === 'evaluated' ? 'Test split' : 'Not measured'}</td>
                      <td>{model.images ?? '—'}</td>
                      {METRIC_COLUMNS.map((metric) => (
                        <td key={metric.key}>{formatMetric(model.metrics?.[metric.key], metric.digits)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {models.filter((model) => model.note).map((model) => (
              <p key={model.name} className="page-description" style={{ margin: '0.75rem 0 0' }}>
                <strong>{model.name}:</strong> {model.note}
              </p>
            ))}
          </div>

          <h3 className="section-title">Test-set metric comparisons</h3>
          <p className="page-description">
            Autoencoder, VAE, and GAN test artifacts cover 828 images. These metrics measure
            reconstruction similarity, not whether a design successfully adds furniture.
            Transformer and Diffusion use different tasks and have no comparable benchmark scores.
          </p>
          <div className="grid-2" style={{ marginBottom: '2rem' }}>
            <ComparisonChartCard title="SSIM" data={chartData('ssim')} />
            <ComparisonChartCard title="PSNR" data={chartData('psnr')} unit=" dB" />
            <ComparisonChartCard title="MSE" data={chartData('mse')} lowerIsBetter />
            <ComparisonChartCard
              title="Mean inference time"
              data={chartData('inference_time_ms')}
              unit=" ms"
              lowerIsBetter
            />
          </div>

          <div className="card">
            <h3 className="section-title" style={{ marginTop: 0 }}>How to interpret these results</h3>
            <p className="page-description">
              SSIM and PSNR are higher-is-better reconstruction measures; MSE and inference time
              are lower-is-better. They are not a single overall quality score, and performance
              should not be ranked across different tasks. A lower reconstruction error can simply
              mean an output looks more like its input; it does not prove that a model furnished a room.
            </p>
            <p className="page-description" style={{ marginBottom: 0 }}>
              Transformer F1 requires labeled expected extractions, and Diffusion quality requires
              an appropriate held-out image-to-image benchmark. Neither score is available in the
              current project artifacts, so no value is fabricated.
            </p>
          </div>
        </>
      )}
    </div>
  );
};

export default Comparison;
