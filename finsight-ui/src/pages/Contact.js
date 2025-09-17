// Contact.js
import React, { useState, useEffect } from "react";
import { FaEnvelope, FaPhone, FaMapMarkerAlt, FaClock } from 'react-icons/fa';

const Contact = () => {
  const [formData, setFormData] = useState({ name: "", email: "", message: "" });
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [expandedFaq, setExpandedFaq] = useState(null);

  const validateEmail = (email) => /\S+@\S+\.\S+/.test(email);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess(false);

    if (!formData.name || !formData.email || !formData.message) {
      setError("⚠️ Please fill in all fields.");
      return;
    }
    if (!validateEmail(formData.email)) {
      setError("⚠️ Please enter a valid email address.");
      return;
    }
    if (formData.message.length < 10) {
      setError("⚠️ Message should be at least 10 characters long.");
      return;
    }

    try {
      setLoading(true);
      await fetch("/api/contact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });
      setSuccess(true);
      setFormData({ name: "", email: "", message: "" });
      setTimeout(() => setSuccess(false), 4000);
    } catch (err) {
      setError("❌ Failed to send message. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const toggleFaq = (index) => {
    setExpandedFaq(expandedFaq === index ? null : index);
  };

  const faqs = [
    {
      question: "How secure is my data?",
      answer: "Your data is encrypted end-to-end and automatically deleted after processing. We comply with GDPR and other standards."
    },
    {
      question: "What formats do you support?",
      answer: "PDF, CSV, Excel, images, and more. Our AI handles scanned documents with OCR."
    },
    {
      question: "How long does analysis take?",
      answer: "Most analyses complete in under 60 seconds, with real-time results for eligibility checks."
    }
  ];

  return (
    <div className="main-content">
      {/* Contact Us Section */}
      <section className="api-section slide-up" id="contact">
        <h2 style={{ textAlign: 'center' }}>Contact Us</h2>
        <p style={{ textAlign: 'center', margin: '1rem auto', maxWidth: 'min(600px, 90%)' }}>Ready to get started? Reach out to our team for demos, support, or custom solutions.</p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '2rem', maxWidth: '900px', margin: '0 auto', padding: '1rem' }}>
          <div style={{ textAlign: 'center', padding: '1.5rem', background: '#f8f9fa', borderRadius: '8px' }}>
            <FaEnvelope className="service-icon" style={{ fontSize: '2.5rem', marginBottom: '1rem', color: '#007bff' }} />
            <h3 style={{ margin: '0.5rem 0' }}>Email</h3>
            <p style={{ margin: '0.5rem 0', fontWeight: '500' }}>hello@finsight.com</p>
          </div>
          <div style={{ textAlign: 'center', padding: '1.5rem', background: '#f8f9fa', borderRadius: '8px' }}>
            <h3 style={{ margin: '0.5rem 0' }}>Phone</h3>
            <p style={{ margin: '0.5rem 0', fontWeight: '500' }}>+91 12345 67890</p>
          </div>
          <div style={{ textAlign: 'center', padding: '1.5rem', background: '#f8f9fa', borderRadius: '8px' }}>
            <h3 style={{ margin: '0.5rem 0' }}>Form</h3>
            <a href="/contact" className="btn btn-primary" style={{ display: 'block', width: '100%', marginTop: '0.5rem', padding: '0.5rem' }}>Send Message</a>
          </div>
        </div>
      </section>

      {/* Hero Section */}
      <div className="hero-grid">
        <div className="hero-card fade-in">
          <div className="landing-tag">Connect with Us</div>
          <h1>Get In Touch</h1>
          <p>Ready to transform your business? Let's discuss how we can help you achieve your goals. Our team is here to answer your questions and explore opportunities.</p>
          <div style={{ display: 'flex', gap: '8px', justifyContent: 'center', flexWrap: 'wrap' }}>
            <button className="btn btn-primary" onClick={() => document.getElementById('contact-form').scrollIntoView({ behavior: 'smooth' })}>Send Message</button>
            <button className="btn btn-secondary" onClick={() => document.getElementById('contact-info').scrollIntoView({ behavior: 'smooth' })}>Contact Details</button>
          </div>
        </div>
      </div>

      {/* Contact Form and Info Side by Side */}
      <section style={{ display: 'flex', flexWrap: 'wrap', gap: '2rem', justifyContent: 'center', margin: '2rem 0', maxWidth: '1200px', marginLeft: 'auto', marginRight: 'auto' }}>
        {/* Form */}
        <div id="contact-form" className="service-card slide-up" style={{ flex: '1 1 400px', maxWidth: '500px', padding: '2rem', marginBottom: '2rem' }}>
          <h3 style={{ color: '#1a1a2e', marginBottom: '1rem' }}>Send a Message</h3>
          <form onSubmit={handleSubmit}>
            <input 
              type="text" 
              placeholder="Full Name" 
              value={formData.name} 
              onChange={(e) => setFormData({ ...formData, name: e.target.value })} 
              required
              style={{ 
                width: '100%', 
                marginBottom: '1rem', 
                padding: '0.75rem', 
                borderRadius: '8px', 
                border: '1px solid #ddd', 
                backgroundColor: '#f9f9f9',
                fontSize: '1rem'
              }}
            />
            <input 
              type="email" 
              placeholder="Email Address" 
              value={formData.email} 
              onChange={(e) => setFormData({ ...formData, email: e.target.value })} 
              required
              style={{ 
                width: '100%', 
                marginBottom: '1rem', 
                padding: '0.75rem', 
                borderRadius: '8px', 
                border: '1px solid #ddd', 
                backgroundColor: '#f9f9f9',
                fontSize: '1rem'
              }}
            />
            <textarea 
              placeholder="Your Message" 
              value={formData.message} 
              onChange={(e) => setFormData({ ...formData, message: e.target.value })} 
              required
              rows={4}
              style={{ 
                width: '100%', 
                marginBottom: '1rem', 
                padding: '0.75rem', 
                borderRadius: '8px', 
                border: '1px solid #ddd', 
                backgroundColor: '#f9f9f9',
                fontSize: '1rem',
                resize: 'vertical'
              }}
            />
            <button 
              type="submit" 
              disabled={loading}
              style={{ 
                width: '100%', 
                padding: '0.75rem', 
                backgroundColor: '#3498db', 
                color: 'white', 
                border: 'none', 
                borderRadius: '8px', 
                fontSize: '1rem', 
                fontWeight: '500',
                cursor: loading ? 'not-allowed' : 'pointer'
              }}
            >
              {loading ? 'Sending...' : 'Send Message'}
            </button>
          </form>
          {success && <p style={{ color: '#27ae60', marginTop: '1rem', textAlign: 'center' }}>✅ Message sent successfully!</p>}
          {error && <p style={{ color: '#e74c3c', marginTop: '1rem', textAlign: 'center' }}>{error}</p>}
        </div>

        {/* Contact Info */}
        <div id="contact-info" className="service-card slide-up" style={{ flex: '1 1 400px', maxWidth: '500px', padding: '2rem', marginBottom: '2rem' }}>
          <h3 style={{ color: '#1a1a2e', marginBottom: '1.5rem' }}>Contact Information</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {/* Email */}
            <div style={{ display: 'flex', alignItems: 'center', padding: '1rem', backgroundColor: '#ecf0f1', borderRadius: '8px', borderLeft: '4px solid #2980b9' }}>
              <FaEnvelope style={{ color: '#2980b9', fontSize: '1.5rem', marginRight: '1rem', flexShrink: 0 }} />
              <div style={{ flex: 1 }}>
                <p style={{ color: '#4a4a67', fontSize: '0.9rem', marginBottom: '0.25rem', fontWeight: '500' }}>Email Us</p>
                <p style={{ color: '#2c3e50', fontSize: '1rem', margin: 0, cursor: 'pointer' }} onClick={() => window.location.href = 'mailto:contact@qfinsight.com'}>contact@qfinsight.com</p>
              </div>
            </div>

            {/* Phone */}
            <div style={{ display: 'flex', alignItems: 'center', padding: '1rem', backgroundColor: '#ecf0f1', borderRadius: '8px', borderLeft: '4px solid #27ae60' }}>
              <FaPhone style={{ color: '#27ae60', fontSize: '1.5rem', marginRight: '1rem', flexShrink: 0 }} />
              <div style={{ flex: 1 }}>
                <p style={{ color: '#4a4a67', fontSize: '0.9rem', marginBottom: '0.25rem', fontWeight: '500' }}>Call Us</p>
                <p style={{ color: '#2c3e50', fontSize: '1rem', margin: 0, cursor: 'pointer' }} onClick={() => window.location.href = 'tel:+15551234567'}>+1 (555) 123-4567</p>
              </div>
            </div>

            {/* Address */}
            <div style={{ display: 'flex', alignItems: 'flex-start', padding: '1rem', backgroundColor: '#ecf0f1', borderRadius: '8px', borderLeft: '4px solid #8e44ad' }}>
              <FaMapMarkerAlt style={{ color: '#8e44ad', fontSize: '1.5rem', marginRight: '1rem', marginTop: '0.25rem', flexShrink: 0 }} />
              <div style={{ flex: 1 }}>
                <p style={{ color: '#4a4a67', fontSize: '0.9rem', marginBottom: '0.25rem', fontWeight: '500' }}>Visit Us</p>
                <p style={{ color: '#2c3e50', fontSize: '1rem', margin: 0 }}>123 Business Ave, Suite 100</p>
                <p style={{ color: '#2c3e50', fontSize: '1rem', margin: '0.25rem 0 0 0' }}>New York, NY 10001</p>
              </div>
            </div>

            {/* Business Hours */}
            <div style={{ display: 'flex', alignItems: 'flex-start', padding: '1rem', backgroundColor: '#ecf0f1', borderRadius: '8px', borderLeft: '4px solid #2980b9' }}>
              <FaClock style={{ color: '#2980b9', fontSize: '1.5rem', marginRight: '1rem', marginTop: '0.25rem', flexShrink: 0 }} />
              <div style={{ flex: 1 }}>
                <p style={{ color: '#4a4a67', fontSize: '0.9rem', marginBottom: '0.25rem', fontWeight: '500' }}>Business Hours</p>
                <ul style={{ color: '#2c3e50', fontSize: '0.95rem', margin: '0', paddingLeft: '1rem' }}>
                  <li style={{ marginBottom: '0.25rem' }}>Mon-Fri: 9:00 AM - 6:00 PM</li>
                  <li style={{ marginBottom: '0.25rem' }}>Sat: 10:00 AM - 4:00 PM</li>
                  <li>Sun: Closed</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FAQ Section - Accordion Layout */}
      <section className="api-section slide-up" style={{ maxWidth: '1200px', margin: '3rem auto', padding: '0 1rem' }}>
        <h2 style={{ color: '#1a1a2e', textAlign: 'center', marginBottom: '2rem' }}>Frequently Asked Questions</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {faqs.map((faq, index) => (
            <div key={index} className="faq-item" style={{ backgroundColor: '#fff', padding: '1.5rem', borderRadius: '8px', boxShadow: '0 4px 8px rgba(0,0,0,0.1)', cursor: 'pointer' }} onClick={() => toggleFaq(index)}>
              <h3 style={{ color: '#1a1a2e', margin: '0', fontSize: '1.2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                {faq.question}
                <span style={{ fontSize: '1.5rem', color: '#3498db' }}>{expandedFaq === index ? '−' : '+'}</span>
              </h3>
              {expandedFaq === index && <p style={{ color: '#4a4a67', marginTop: '0.5rem' }}>{faq.answer}</p>}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};

export default Contact;