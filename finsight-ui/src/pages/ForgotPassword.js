// src/pages/ForgotPassword.js
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { FaEnvelope } from 'react-icons/fa';
import supabase from '../utils/supabaseClient';

const ForgotPassword = () => {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setMessage('');

    if (!email.trim().match(/[^@]+@[^@]+\.[^@]+/)) {
      setError('Invalid email format');
      setLoading(false);
      return;
    }

    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email.trim(), {
        redirectTo: `${window.location.origin}/reset-password`
      });

      if (error) {
        setError('Failed to send reset email. Please check your email address.');
      } else {
        setMessage('Password reset email sent! Check your inbox.');
      }
    } catch (err) {
      setError('An error occurred. Please try again.');
      console.error('Forgot Password Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center',  minHeight: '50vh', minWidth: '85vh', background: 'linear-gradient(135deg, #e6f0fa 0%, #f8f9fa 100%)', padding: '40px 20px' }}>
      <div style={{ maxWidth: '600px', width: '100%', background: '#fff', borderRadius: '15px', boxShadow: '0 6px 20px rgba(0,0,0,0.08)', padding: '40px' }}>
        <h2 style={{ color: '#2c3e50', textAlign: 'center', marginBottom: '30px', fontSize: '2rem', fontWeight: 700 }}>Reset Your Password</h2>
        {error && (
          <div style={{ color: '#e74c3c', textAlign: 'center', marginBottom: '20px', fontSize: '0.95rem', background: '#fee', padding: '15px', borderRadius: '6px', border: '1px solid #fcc' }}>
            {error}
          </div>
        )}
        {message && (
          <div style={{ color: '#2ecc71', textAlign: 'center', marginBottom: '20px', fontSize: '0.95rem', background: '#e8f5e8', padding: '15px', borderRadius: '6px', border: '1px solid #ceface' }}>
            {message}
          </div>
        )}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div>
            <label style={{ display: 'block', color: '#3498db', fontWeight: 600, marginBottom: '10px' }}><FaEnvelope style={{ marginRight: '10px' }} /> Email Address</label>
            <input 
              type="email" 
              placeholder="Enter your email" 
              value={email} 
              onChange={(e) => setEmail(e.target.value)} 
              style={{ width: '100%', padding: '12px', border: '1px solid #ddd', borderRadius: '6px', fontSize: '1rem', background: '#f9f9f9' }}
              required 
            />
          </div>
          <button 
            type="submit" 
            disabled={loading}
            style={{ padding: '12px', background: loading ? '#bdc3c7' : 'linear-gradient(90deg, #2c3e50, #3498db)', border: 'none', borderRadius: '6px', color: 'white', fontWeight: 600, fontSize: '1.1rem', cursor: loading ? 'not-allowed' : 'pointer', transition: 'transform 0.2s' }}
          >
            {loading ? 'Sending...' : 'Send Reset Link'}
          </button>
        </form>
        <p style={{ textAlign: 'center', marginTop: '20px', color: '#666', fontSize: '0.95rem' }}>
          Back to <Link to="/login" style={{ color: '#3498db', textDecoration: 'none', fontWeight: 500 }}>Login</Link>
        </p>
      </div>
    </div>
  );
};

export default ForgotPassword;