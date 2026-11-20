/**
 * API client for the FastAPI backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

async function parseError(response) {
  if (response.status === 404) {
    return 'Backend route not found. Restart the FastAPI server after updating backend code.';
  }
  try {
    const payload = await response.json();
    const detail = payload.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail.map((item) => item.msg || JSON.stringify(item)).join(' ');
    }
    if (detail) return JSON.stringify(detail);
  } catch {
    /* fall through */
  }
  if (response.status === 502 || response.status === 503) {
    return 'Backend unavailable or the requested model checkpoint is missing.';
  }
  return `Request failed (${response.status}).`;
}

export const analyzeDesignPrompt = async (prompt) => {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/transformer/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
  } catch {
    throw new Error('Backend unavailable. Start the FastAPI server and try again.');
  }
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
};

export const getModelComparison = async () => {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/comparison/metrics`);
  } catch {
    throw new Error('Backend unavailable. Start the FastAPI server and try again.');
  }
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
};

export const generateDesign = async (formData) => {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/autoencoder/predict`, {
      method: 'POST',
      body: formData,
    });
  } catch {
    throw new Error('Backend unavailable. Start the FastAPI server and try again.');
  }
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json();
};

export const generateInteriorImage = async (design) => {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/generation/image`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(design),
    });
  } catch {
    throw new Error('Backend unavailable. Start the FastAPI server and try again.');
  }
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json();
};

export const generateDiffusionImage = async (formData) => {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/generation/diffuse`, {
      method: 'POST',
      body: formData,
    });
  } catch {
    throw new Error('Backend unavailable. Start the FastAPI server and try again.');
  }
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json();
};

export const getAutoencoderHistory = async () => {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/autoencoder/history`);
  } catch {
    return [];
  }
  if (!response.ok) return [];
  const payload = await response.json();
  return payload.data || [];
};

export const getVaeStatus = async () => {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/vae/status`);
  } catch {
    throw new Error('Backend unavailable. Start the FastAPI server and try again.');
  }
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
};

export const getVaeHistory = async () => {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/vae/history`);
  } catch {
    throw new Error('Backend unavailable. Start the FastAPI server and try again.');
  }
  if (!response.ok) throw new Error(await parseError(response));
  const payload = await response.json();
  return payload.data || [];
};

export const reconstructVae = async (formData) => {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/vae/reconstruct`, {
      method: 'POST',
      body: formData,
    });
  } catch {
    throw new Error('Backend unavailable. Start the FastAPI server and try again.');
  }
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
};

export const generateVaeSample = async (seed = null) => {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/vae/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ seed }),
    });
  } catch {
    throw new Error('Backend unavailable. Start the FastAPI server and try again.');
  }
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
};

export const getGanStatus = async () => {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/gan/status`);
  } catch {
    throw new Error('Backend unavailable. Start the FastAPI server and try again.');
  }
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
};

export const getGanHistory = async () => {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/gan/history`);
  } catch {
    return [];
  }
  if (!response.ok) return [];
  const payload = await response.json();
  return payload.data || [];
};

export const generateGan = async (formData) => {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/gan/generate`, {
      method: 'POST',
      body: formData,
    });
  } catch {
    throw new Error('Backend unavailable. Start the FastAPI server and try again.');
  }
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
};

export const getModelMetrics = async (modelName) => {
  console.log(`getModelMetrics called for: ${modelName}`);
  return null;
};

export const getAllMetrics = async () => {
  return [];
};
