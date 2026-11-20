import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_');
  const backendTarget = env.VITE_BACKEND_URL || 'http://127.0.0.1:8001';

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': backendTarget,
        '/results': backendTarget,
      },
    },
  };
});
