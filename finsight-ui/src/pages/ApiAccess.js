// ApiAccess.js
import React from 'react';
import { motion } from 'framer-motion';

const ApiAccess = () => {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
      <div className="main-content">
        {/* API Access for Developers Section */}
        <section className="api-section fade-in" id="api-access">
          <h2>API Access for Developers</h2>
          <p>Integrate FinSight's powerful AI capabilities into your applications. Visit our <a href="https://x.ai/api" target="_blank" rel="noopener noreferrer">API documentation</a> for details on endpoints, authentication, and usage.</p>
          <p>Unlock custom financial solutions with seamless API integration—perfect for fintech innovators.</p>
        </section>

        {/* Hero for API */}
        <div className="hero-grid">
          <div className="hero-card fade-in">
            <div className="landing-tag">Developer Resources</div>
            <h1>Integrate with FinSight API</h1>
            <p>Embed powerful AI-driven financial analysis into your applications seamlessly. Our API provides secure, real-time insights for loan decisions, document processing, and more.</p>
            <div style={{ display: 'flex', gap: '8px', justifyContent: 'center', flexWrap: 'wrap' }}>
              <a href="#docs" className="btn btn-primary">View Docs</a>
              <a href="/register" className="btn btn-secondary">Get API Key</a>
            </div>
          </div>
        </div>

        {/* API Features Section - Side by Side Layout */}
        <section className="features" id="features" style={{ display: 'flex', flexWrap: 'wrap', gap: '2rem', justifyContent: 'center', maxWidth: '1200px', margin: '0 auto' }}>
          <div className="service-card slide-up" style={{ flex: '1 1 45%', minWidth: '300px', maxWidth: '500px', padding: '2rem', marginBottom: '2rem' }}>
            <h3 style={{ color: '#1a1a2e', marginBottom: '1rem' }}>Document Upload API</h3>
            <p style={{ color: '#4a4a67', marginBottom: '0.5rem' }}><strong>Endpoint:</strong> POST /api/upload</p>
            <p style={{ color: '#4a4a67', marginBottom: '1rem' }}>Upload bank statements or financial documents for instant analysis. Supports PDF, CSV, and image formats with OCR.</p>
            <ul style={{ color: '#4a4a67', listStyleType: 'disc', paddingLeft: '1.5rem', marginBottom: '1rem' }}>
              <li style={{ marginBottom: '0.5rem' }}>Multi-file support</li>
              <li style={{ marginBottom: '0.5rem' }}>Automatic data extraction</li>
              <li style={{ marginBottom: '0.5rem' }}>Secure temporary storage</li>
            </ul>
            <a href="#docs" className="learn-more" style={{ color: '#3498db', textDecoration: 'none', fontWeight: '500' }}>Learn More →</a>
          </div>

          <div className="service-card slide-up" style={{ flex: '1 1 45%', minWidth: '300px', maxWidth: '500px', padding: '2rem', marginBottom: '2rem', animationDelay: '0.2s' }}>
            <h3 style={{ color: '#1a1a2e', marginBottom: '1rem' }}>Loan Eligibility Check</h3>
            <p style={{ color: '#4a4a67', marginBottom: '0.5rem' }}><strong>Endpoint:</strong> POST /api/eligibility</p>
            <p style={{ color: '#4a4a67', marginBottom: '1rem' }}>Evaluate loan eligibility based on financial data, CIBIL score, and custom parameters. Returns detailed recommendations.</p>
            <ul style={{ color: '#4a4a67', listStyleType: 'disc', paddingLeft: '1.5rem', marginBottom: '1rem' }}>
              <li style={{ marginBottom: '0.5rem' }}>Real-time scoring</li>
              <li style={{ marginBottom: '0.5rem' }}>Customizable thresholds</li>
              <li style={{ marginBottom: '0.5rem' }}>Risk assessment metrics</li>
            </ul>
            <a href="#docs" className="learn-more" style={{ color: '#3498db', textDecoration: 'none', fontWeight: '500' }}>Learn More →</a>
          </div>

          <div className="service-card slide-up" style={{ flex: '1 1 45%', minWidth: '300px', maxWidth: '500px', padding: '2rem', marginBottom: '2rem', animationDelay: '0.4s' }}>
            <h3 style={{ color: '#1a1a2e', marginBottom: '1rem' }}>Report Generation</h3>
            <p style={{ color: '#4a4a67', marginBottom: '0.5rem' }}><strong>Endpoint:</strong> GET /api/reports/reportId</p>
            <p style={{ color: '#4a4a67', marginBottom: '1rem' }}>Generate comprehensive PDF reports with visualizations, summaries, and actionable insights from analyzed data.</p>
            <ul style={{ color: '#4a4a67', listStyleType: 'disc', paddingLeft: '1.5rem', marginBottom: '1rem' }}>
              <li style={{ marginBottom: '0.5rem' }}>Custom templates</li>
              <li style={{ marginBottom: '0.5rem' }}>Export options (PDF, CSV)</li>
              <li style={{ marginBottom: '0.5rem' }}>Branded reports</li>
            </ul>
            <a href="#docs" className="learn-more" style={{ color: '#3498db', textDecoration: 'none', fontWeight: '500' }}>Learn More →</a>
          </div>

          <div className="service-card slide-up" style={{ flex: '1 1 45%', minWidth: '300px', maxWidth: '500px', padding: '2rem', marginBottom: '2rem', animationDelay: '0.6s' }}>
            <h3 style={{ color: '#1a1a2e', marginBottom: '1rem' }}>Authentication & Security</h3>
            <p style={{ color: '#4a4a67', marginBottom: '0.5rem' }}><strong>Features:</strong> OAuth 2.0, API Keys</p>
            <p style={{ color: '#4a4a67', marginBottom: '1rem' }}>Ensure secure access with robust authentication methods. All data is encrypted in transit and at rest.</p>
            <ul style={{ color: '#4a4a67', listStyleType: 'disc', paddingLeft: '1.5rem', marginBottom: '1rem' }}>
              <li style={{ marginBottom: '0.5rem' }}>Rate limiting</li>
              <li style={{ marginBottom: '0.5rem' }}>IP whitelisting</li>
              <li style={{ marginBottom: '0.5rem' }}>Audit logs</li>
            </ul>
            <a href="#docs" className="learn-more" style={{ color: '#3498db', textDecoration: 'none', fontWeight: '500' }}>Learn More →</a>
          </div>
        </section>

        {/* Benefits Section */}
        <section className="api-section slide-up">
          <h2>Why Use Our API?</h2>
          <p>Scale your financial services with reliable, AI-powered tools. Integrate easily with SDKs for Node.js, Python, and more.</p>
          <div className="features" style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap', justifyContent: 'center' }}>
            <div className="service-card" style={{ flex: '1 1 300px' }}>
              <h3>Scalability</h3>
              <p>Handle high volumes with auto-scaling infrastructure. Pay only for what you use.</p>
            </div>
            <div className="service-card" style={{ flex: '1 1 300px' }}>
              <h3>Accuracy</h3>
              <p>99%+ accuracy in data extraction and analysis, powered by advanced ML models.</p>
            </div>
            <div className="service-card" style={{ flex: '1 1 300px' }}>
              <h3>Support</h3>
              <p>24/7 developer support, detailed documentation, and community forums.</p>
            </div>
          </div>
        </section>
      </div>
    </motion.div>
  );
};

export default ApiAccess;