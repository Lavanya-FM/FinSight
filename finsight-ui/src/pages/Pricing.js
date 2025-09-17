// Pricing.js
import React from 'react';

const Pricing = () => {
  return (
    <div className="main-content">
      {/* Hero for Pricing */}
      <div className="hero-grid">
        <div className="hero-card fade-in">
          <div className="landing-tag">Choose Your Plan</div>
          <h1>Flexible Pricing</h1>
          <p>Select the perfect plan for your needs. All plans include core features with scalable options for growth.</p>
        </div>
      </div>

      {/* Pricing Plans */}
      <section className="features" id="pricing">
        <div className="service-card slide-up">
          <h3>Starter</h3>
          <p><strong>Free</strong></p>
          <ul>
            <li>5 API calls/day</li>
            <li>Basic document analysis</li>
            <li>Email support (48h response)</li>
            <li>Standard security features</li>
            <li>Community forum access</li>
          </ul>
          <a href="/register" className="btn btn-primary">Try Free</a>
        </div>

        <div className="service-card slide-up" style={{ animationDelay: '0.2s' }}>
          <h3>Pro</h3>
          <p><strong>₹999/mo</strong></p>
          <ul>
            <li>500 API calls/month</li>
            <li>Advanced analysis with insights</li>
            <li>Priority email support (24h)</li>
            <li>Custom report templates</li>
            <li>API usage analytics</li>
          </ul>
          <a href="/pricing" className="btn btn-primary">Upgrade</a>
        </div>

        <div className="service-card slide-up" style={{ animationDelay: '0.4s' }}>
          <h3>Enterprise</h3>
          <p><strong>Custom</strong></p>
          <ul>
            <li>Unlimited API calls</li>
            <li>Custom AI models</li>
            <li>Dedicated support team</li>
            <li>On-premise deployment option</li>
            <li>SLAs and uptime guarantees</li>
          </ul>
          <a href="/contact" className="btn btn-primary">Contact Us</a>
        </div>
      </section>

      {/* Additional Info */}
      <section className="api-section slide-up">
        <h2>Billing Details</h2>
        <p>All plans are billed monthly. Cancel anytime. Enterprise plans include custom features and dedicated resources.</p>
        <div style={{ display: 'flex', gap: '2rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <div className="service-card" style={{ flex: '1 1 300px' }}>
            <h3>Payment Methods</h3>
            <p>We accept credit cards, PayPal, and bank transfers for enterprise.</p>
          </div>
          <div className="service-card" style={{ flex: '1 1 300px' }}>
            <h3>Refund Policy</h3>
            <p>30-day money-back guarantee on Pro plans.</p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Pricing;