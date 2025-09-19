import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'https://riskiq.onrender.com',
        changeOrigin: true,
        secure: true,
        // FIX: No rewrite needed if backend is /api/v1 (e.g., /api/v1/health -> https://riskiq.onrender.com/api/v1/health)
        // If backend root is /v1, uncomment: rewrite: (path) => path.replace(/^\/api/, '/v1'),
        configure: (proxy, options) => {
          // LOG: Debug proxy triggers
          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log('[VITE PROXY] Forwarding:', req.url, 'to', proxyReq.getHeader('host') + proxyReq.path);
          });
          proxy.on('error', (err, req, res) => {
            console.error('[VITE PROXY ERROR]', err, 'for', req.url);
          });
          proxy.on('proxyRes', (proxyRes, req, res) => {
            console.log('[VITE PROXY] Response from backend:', proxyRes.statusCode, 'for', req.url);
          });
        },
      },
    },
  },
});