// src/pages/Dashboard/HelpAndSupport.js
import React from 'react';

const HelpAndSupport = () => {
  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '40px 20px' }}>
      <h1 style={{ color: '#2c3e50', marginBottom: '30px', fontSize: '2rem', fontWeight: 700 }}>Help & Support</h1>
      <div style={{ background: '#fff', borderRadius: '10px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', padding: '30px' }}>
        <h3 style={{ color: '#2c3e50', marginBottom: '15px' }}>Getting Started</h3>
        <p style={{ color: '#666', marginBottom: '20px' }}>Upload your bank statements and enter your CIBIL score to get AI-powered insights.</p>
        
        <h3 style={{ color: '#2c3e50', marginBottom: '15px' }}>FAQ</h3>
        <ul style={{ paddingLeft: '20px', color: '#666', marginBottom: '20px' }}>
          <li><strong style={{ color: '#3498db' }}>What file formats are supported?</strong> PDF bank statements.</li>
          <li><strong style={{ color: '#3498db' }}>How accurate is the analysis?</strong> Our ML model is trained on anonymized data for high accuracy.</li>
          <li><strong style={{ color: '#3498db' }}>Privacy?</strong> All data is processed securely and not stored beyond analysis.</li>
        </ul>
        
        <h3 style={{ color: '#2c3e50', marginBottom: '15px' }}>Contact Us</h3>
        <p style={{ color: '#666', marginBottom: '10px' }}>Email: support@finsight.com</p>
        <p style={{ color: '#666' }}>Phone: +1-234-567-8900</p>
      </div>
    </div>
  );
};

export default HelpAndSupport;