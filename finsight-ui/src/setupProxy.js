const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  // Get target URL from environment variables or use default
  const target = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  console.log('Setting up proxy to:', target);

  app.use(
    '/v1',
    createProxyMiddleware({
      target: target,
      changeOrigin: true,
      pathRewrite: {
        '^/v1': '/api/v1'
      },
      onProxyReq: (proxyReq, req, res) => {
        console.log('[CRA PROXY] Forwarding:', req.url, 'to', proxyReq.getHeader('host') + proxyReq.path);
      },
      onError: (err, req, res) => {
        console.error('[CRA PROXY ERROR]', err, 'for', req.url);
        res.writeHead(500, {
          'Content-Type': 'text/plain',
        });
        res.end('Something went wrong. And we are reporting a custom error message.');
      },
      onProxyRes: (proxyRes, req, res) => {
        console.log('[CRA PROXY] Response from backend:', proxyRes.statusCode, 'for', req.url);
      }
    })
  );

  // Health check endpoint for debugging
  app.get('/api/health', (req, res) => {
    res.json({
      status: 'ok',
      environment: process.env.NODE_ENV,
      timestamp: new Date().toISOString(),
      proxyTarget: target
    });
  });
};
