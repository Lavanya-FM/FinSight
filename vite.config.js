import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/v1': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/v1/, '/api/v1'),
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