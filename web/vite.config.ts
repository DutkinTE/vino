import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
  },
  server: {
    proxy: {
      '/predict': 'http://127.0.0.1:8080',
      '/health': 'http://127.0.0.1:8080',
    },
  },
});
