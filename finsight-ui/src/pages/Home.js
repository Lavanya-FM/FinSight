// Home.js
import React, { useEffect } from 'react';
import { FaFileUpload, FaDollarSign, FaShieldAlt, FaChartLine, FaEnvelope } from 'react-icons/fa';
import { Link } from 'react-router-dom';

const Home = () => {
  // Update date/time dynamically
  useEffect(() => {
    const updateTime = () => {
      const dateTimeElement = document.getElementById('current-date-time');
      if (dateTimeElement) {
        const now = new Date();
        dateTimeElement.textContent = `Current Date and Time: ${now.toLocaleString('en-IN', {
          timeZone: 'Asia/Kolkata',
          hour12: true,
          weekday: 'long',
          year: 'numeric',
          month: 'long',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        })} IST`;
      }
    };
    updateTime(); // Initial call (should show ~03:06 PM IST on Sep 11, 2025)
    const interval = setInterval(updateTime, 60000); // Update every minute
    return () => clearInterval(interval); // Cleanup on unmount
  }, []);

  // JSON as string for JSX
  const apiResponse = JSON.stringify({
    "eligibility": "Approved",
    "loan_limit": 850000,
    "recommendation": "Offer EMI of ₹19,500 for 48 months"
  }, null, 2);

  return (
    <div className="main-content">
      {/* Hero Section */}
      <div className="hero-grid">
        <div className="hero-card fade-in">
          <div className="landing-tag">AI-Powered Financial Edge</div>
          <h1>Elevate Your Loan Strategy</h1>
          <p>Unlock real-time insights with secure document analysis, tailored for financial experts and businesses.</p>
          <div style={{ display: 'flex', gap: '8px', justifyContent: 'center', flexWrap: 'wrap' }}>
            <Link to="/register" className="btn btn-primary">Get Started</Link>
            <Link to="/login" className="btn btn-secondary">Access Now</Link>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <section className="features" id="features">
        <div className="service-card slide-up">
          <FaFileUpload className="service-icon" />
          <h3>DocSync Mastery</h3>
          <p><strong>Seamless Upload Experience</strong></p>
          <ul>
            <li>Multi-format support (PDFs, CSVs)</li>
            <li>Smart OCR for scanned documents</li>
            <li>Instant transaction extraction</li>
          </ul>
          <a href="#features" className="learn-more">Explore →</a>
        </div>

        <div className="service-card slide-up" style={{ animationDelay: '0.2s' }}>
          <FaDollarSign className="service-icon" />
          <h3>FinPulse Engine</h3>
          <p><strong>Smart Financial Tools</strong></p>
          <ul>
            <li>Custom EMI optimization</li>
            <li>Real-time credit score insights</li>
            <li>Advanced risk evaluation</li>
          </ul>
          <a href="#features" className="learn-more">Learn More →</a>
        </div>

        <div className="service-card slide-up" style={{ animationDelay: '0.4s' }}>
          <FaShieldAlt className="service-icon" />
          <h3>SecureVault Protection</h3>
          <p><strong>Uncompromising Security</strong></p>
          <ul>
            <li>End-to-end encryption</li>
            <li>GDPR and compliance standards</li>
            <li>No data retention guaranteed</li>
          </ul>
          <a href="#features" className="learn-more">See Security →</a>
        </div>

        <div className="service-card slide-up" style={{ animationDelay: '0.6s' }}>
          <FaChartLine className="service-icon" />
          <h3>InsightFlow Analytics</h3>
          <p><strong>Dynamic Data Insights</strong></p>
          <ul>
            <li>Interactive financial dashboards</li>
            <li>Predictive trend forecasting</li>
            <li>Exportable detailed reports</li>
          </ul>
          <a href="#features" className="learn-more">View Insights →</a>
        </div>
      </section>

      {/* Developer API Access Section */}
      <section className="api-section slide-up">
        <h2>Developer API Access</h2>
        <p>Use our REST API to embed loan decision intelligence into your apps or systems.</p>
        <div className="service-card" style={{ textAlign: 'left', padding: '1.5rem' }}>
          <h3>API Endpoint: Analyze Loan Data</h3>
          <p><strong>Method:</strong> POST /api/analyze</p>
          <p><strong>Headers:</strong></p>
          <ul>
            <li>Authorization: Bearer &lt;YOUR_API_KEY&gt;</li>
            <li>Content-Type: multipart/form-data</li>
          </ul>
          <p><strong>Body:</strong></p>
          <ul>
            <li>file: &lt;bank_statement.pdf&gt;</li>
          </ul>
          <p><strong>Response (JSON):</strong></p>
          <pre style={{ background: '#f8f9fa', padding: '1rem', borderRadius: '6px', margin: '1rem 0' }}>
            {apiResponse}
          </pre>
          <a href="https://x.ai/api" target="_blank" rel="noopener noreferrer" className="learn-more">View Full API Docs →</a>
        </div>
      </section>

      {/* Flexible Pricing Plans Section */}
      <section className="api-section slide-up" id="pricing">
        <h2>Flexible Pricing Plans</h2>
        <div className="features" style={{ gap: '2rem', display: 'flex', justifyContent: 'center' }}>
          <div className="service-card slide-up" style={{ animationDelay: '0.2s', padding: '1.5rem', textAlign: 'center', minWidth: '250px', maxWidth: '300px' }}>
            <h3>Starter</h3>
            <p><strong>Free</strong></p>
            <ul style={{ paddingLeft: '0', listStyle: 'none', textAlign: 'left', margin: '1rem 0' }}>
              <li style={{ marginBottom: '0.5rem' }}>5 API calls/day</li>
              <li style={{ marginBottom: '0.5rem' }}>Email support</li>
              <li style={{ marginBottom: '0.5rem' }}>Basic analysis</li>
            </ul>
            <Link to="/register" className="btn btn-primary" style={{ display: 'block', width: '100%', marginTop: '1rem' }}>Try Free</Link>
          </div>
          <div className="service-card slide-up" style={{ animationDelay: '0.4s', padding: '1.5rem', textAlign: 'center', minWidth: '250px', maxWidth: '300px' }}>
            <h3>Pro</h3>
            <p><strong>₹999/mo</strong></p>
            <ul style={{ paddingLeft: '0', listStyle: 'none', textAlign: 'left', margin: '1rem 0' }}>
              <li style={{ marginBottom: '0.5rem' }}>500 API calls/month</li>
              <li style={{ marginBottom: '0.5rem' }}>Priority support</li>
              <li style={{ marginBottom: '0.5rem' }}>Detailed reports</li>
            </ul>
            <Link to="/pricing" className="btn btn-primary" style={{ display: 'block', width: '100%', marginTop: '1rem' }}>Upgrade</Link>
          </div>
          <div className="service-card slide-up" style={{ animationDelay: '0.6s', padding: '1.5rem', textAlign: 'center', minWidth: '250px', maxWidth: '300px' }}>
            <h3>Enterprise</h3>
            <p><strong>Custom</strong></p>
            <ul style={{ paddingLeft: '0', listStyle: 'none', textAlign: 'left', margin: '1rem 0' }}>
              <li style={{ marginBottom: '0.5rem' }}>Unlimited API calls</li>
              <li style={{ marginBottom: '0.5rem' }}>Dedicated support</li>
              <li style={{ marginBottom: '0.5rem' }}>Custom integrations</li>
            </ul>
            <Link to="/contact" className="btn btn-primary" style={{ display: 'block', width: '100%', marginTop: '1rem' }}>Contact Us</Link>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="testimonials">
        <h2 className="fade-in">Voices of Success</h2>
        <div className="testimonial-card slide-up" style={{ animationDelay: '0.2s' }}>
          <p>"FinSight's precision has redefined our loan approvals—truly transformative."</p>
          <span>- Priya Sharma, Finance Lead, TechCorp</span>
        </div>
        <div className="testimonial-card slide-up" style={{ animationDelay: '0.4s' }}>
          <p>"Speed and security that set a new standard for fintech solutions."</p>
          <span>- Rajesh Kumar, CEO, FinSecure</span>
        </div>
      </section>
    </div>
  );
};

export default Home;