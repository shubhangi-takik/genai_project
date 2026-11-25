const BACKEND_DISABLED_MESSAGE = 'Backend integration is disabled in this frontend.';

const backendDisabled = async () => {
  throw new Error(BACKEND_DISABLED_MESSAGE);
};

export const analyzeDesignPrompt = backendDisabled;
export const getModelComparison = backendDisabled;
export const generateDesign = backendDisabled;
export const generateInteriorImage = backendDisabled;
export const generateDiffusionImage = backendDisabled;

export const getAutoencoderHistory = async () => [];
export const getVaeStatus = async () => ({ available: false, device: null });
export const getVaeHistory = async () => [];
export const reconstructVae = backendDisabled;
export const generateVaeSample = backendDisabled;

export const getGanStatus = async () => ({ available: false, device: null });
export const getGanHistory = async () => [];
export const generateGan = backendDisabled;

export const getModelMetrics = async () => null;
export const getAllMetrics = async () => [];
