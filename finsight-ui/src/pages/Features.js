// Features.js
import React from 'react';
import { FaFileUpload, FaDollarSign, FaShieldAlt, FaChartLine } from 'react-icons/fa';

const Features = () => {
  return (
    <div className="main-content">
      {/* Hero for Features */}
      <div className="hero-grid">
        <div className="hero-card fade-in">
          <div className="landing-tag">Core Capabilities</div>
          <h1>Discover Our Features</h1>
          <p>Explore the powerful tools that make FinSight the leading platform for financial analysis and loan management.</p>
        </div>
      </div>

      {/* Features Cards - Side by Side Layout */}
      <section className="features" id="features" style={{ display: 'flex', flexWrap: 'wrap', gap: '2rem', justifyContent: 'center', maxWidth: '1200px', margin: '0 auto', padding: '2rem 0' }}>
        <div className="service-card slide-up" style={{ flex: '1 1 45%', minWidth: '300px', maxWidth: '500px', padding: '2rem', marginBottom: '2rem' }}>
          <FaFileUpload className="service-icon" />
          <h3 style={{ color: '#1a1a2e', marginBottom: '1rem' }}>DocSync Mastery</h3>
          <p style={{ color: '#4a4a67', marginBottom: '0.5rem' }}><strong>Seamless Upload Experience</strong></p>
          <p style={{ color: '#4a4a67', marginBottom: '1rem' }}>Effortlessly upload and process financial documents with advanced AI recognition.</p>
          <ul style={{ color: '#4a4a67', listStyleType: 'disc', paddingLeft: '1.5rem', marginBottom: '1rem' }}>
            <li style={{ marginBottom: '0.5rem' }}>Multi-format support (PDFs, CSVs, Excel, Images)</li>
            <li style={{ marginBottom: '0.5rem' }}>Smart OCR for scanned documents with 99% accuracy</li>
            <li style={{ marginBottom: '0.5rem' }}>Instant transaction extraction and categorization</li>
            <li style={{ marginBottom: '0.5rem' }}>Batch processing for multiple files</li>
          </ul>
          <a href="#" onClick={() => handleLearnMore('docsync')} className="learn-more" style={{ color: '#3498db', textDecoration: 'none', fontWeight: '500' }}>Explore →</a>
        </div>

        <div className="service-card slide-up" style={{ flex: '1 1 45%', minWidth: '300px', maxWidth: '500px', padding: '2rem', marginBottom: '2rem', animationDelay: '0.2s' }}>
          <FaDollarSign className="service-icon" />
          <h3 style={{ color: '#1a1a2e', marginBottom: '1rem' }}>FinPulse Engine</h3>
          <p style={{ color: '#4a4a67', marginBottom: '0.5rem' }}><strong>Smart Financial Tools</strong></p>
          <p style={{ color: '#4a4a67', marginBottom: '1rem' }}>Intelligent algorithms for comprehensive financial evaluation and decision-making.</p>
          <ul style={{ color: '#4a4a67', listStyleType: 'disc', paddingLeft: '1.5rem', marginBottom: '1rem' }}>
            <li style={{ marginBottom: '0.5rem' }}>Custom EMI optimization with multiple scenarios</li>
            <li style={{ marginBottom: '0.5rem' }}>Real-time credit score insights and predictions</li>
            <li style={{ marginBottom: '0.5rem' }}>Advanced risk evaluation using ML models</li>
            <li style={{ marginBottom: '0.5rem' }}>Automated loan recommendation system</li>
          </ul>
          <a href="#" onClick={() => handleLearnMore('finpulse')} className="learn-more" style={{ color: '#3498db', textDecoration: 'none', fontWeight: '500' }}>Learn More →</a>
        </div>

        <div className="service-card slide-up" style={{ flex: '1 1 45%', minWidth: '300px', maxWidth: '500px', padding: '2rem', marginBottom: '2rem', animationDelay: '0.4s' }}>
          <FaShieldAlt className="service-icon" />
          <h3 style={{ color: '#1a1a2e', marginBottom: '1rem' }}>SecureVault Protection</h3>
          <p style={{ color: '#4a4a67', marginBottom: '0.5rem' }}><strong>Uncompromising Security</strong></p>
          <p style={{ color: '#4a4a67', marginBottom: '1rem' }}>Enterprise-grade security to protect sensitive financial data.</p>
          <ul style={{ color: '#4a4a67', listStyleType: 'disc', paddingLeft: '1.5rem', marginBottom: '1rem' }}>
            <li style={{ marginBottom: '0.5rem' }}>End-to-end encryption (AES-256)</li>
            <li style={{ marginBottom: '0.5rem' }}>GDPR, PCI-DSS, and ISO 27001 compliance</li>
            <li style={{ marginBottom: '0.5rem' }}>No data retention policy with auto-deletion</li>
            <li style={{ marginBottom: '0.5rem' }}>Multi-factor authentication and access controls</li>
          </ul>
          <a href="#" onClick={() => handleLearnMore('securevault')} className="learn-more" style={{ color: '#3498db', textDecoration: 'none', fontWeight: '500' }}>See Security →</a>
        </div>

        <div className="service-card slide-up" style={{ flex: '1 1 45%', minWidth: '300px', maxWidth: '500px', padding: '2rem', marginBottom: '2rem', animationDelay: '0.6s' }}>
          <FaChartLine className="service-icon" />
          <h3 style={{ color: '#1a1a2e', marginBottom: '1rem' }}>InsightFlow Analytics</h3>
          <p style={{ color: '#4a4a67', marginBottom: '0.5rem' }}><strong>Dynamic Data Insights</strong></p>
          <p style={{ color: '#4a4a67', marginBottom: '1rem' }}>Visualize and understand financial data like never before.</p>
          <ul style={{ color: '#4a4a67', listStyleType: 'disc', paddingLeft: '1.5rem', marginBottom: '1rem' }}>
            <li style={{ marginBottom: '0.5rem' }}>Interactive financial dashboards with real-time updates</li>
            <li style={{ marginBottom: '0.5rem' }}>Predictive trend forecasting using time-series analysis</li>
            <li style={{ marginBottom: '0.5rem' }}>Exportable detailed reports in multiple formats</li>
            <li style={{ marginBottom: '0.5rem' }}>Customizable visualizations and KPI tracking</li>
          </ul>
          <a href="#" onClick={() => handleLearnMore('insightflow')} className="learn-more" style={{ color: '#3498db', textDecoration: 'none', fontWeight: '500' }}>View Insights →</a>
        </div>
      </section>

      {/* Additional Features */}
      <section className="api-section slide-up">
        <h2>Advanced Capabilities</h2>
        <p>Beyond the basics, FinSight offers cutting-edge features for power users.</p>
        <div className="features" style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
          <div className="service-card" style={{ flex: '1 1 400px' }}>
            <h3>Integration Ecosystem</h3>
            <p>Seamless integration with popular tools and platforms like Zapier, Salesforce, and banking APIs.</p>
          </div>
          <div className="service-card" style={{ flex: '1 1 400px' }}>
            <h3>AI Customization</h3>
            <p>Tailor AI models to your specific industry needs with our fine-tuning service.</p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Features;