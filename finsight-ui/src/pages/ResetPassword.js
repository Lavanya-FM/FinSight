// src/pages/ResetPassword.js
import React, { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { FaLock } from 'react-icons/fa';
import supabase from '../utils/supabaseClient';

const ResetPassword = () => {
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const token = params.get('access_token');
    if (!token) {
      setError('Invalid or missing reset token. Please request a new reset link.');
    }
  }, [location.search]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setMessage('');

    if (password.length < 8 || !/[!@#$%^&*()_+]/.test(password)) {
      setError('Password must be at least 8 characters and include a special character');
      setLoading(false);
      return;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }

    try {
      const { error } = await supabase.auth.updateUser({ password: password.trim() });

      if (error) {
        setError('Failed to reset password: ' + error.message);
      } else {
        setMessage('Password reset successfully! Redirecting to login...');
        setTimeout(() => navigate('/login'), 2000); // Redirect after 2 seconds
      }
    } catch (err) {
      setError('An error occurred. Please try again.');
      console.error('Reset Password Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center',  minHeight: '50vh', minWidth: '85vh', background: 'linear-gradient(135deg, #e6f0fa 0%, #f8f9fa 100%)', padding: '40px 20px' }}>
      <div style={{ maxWidth: '600px', width: '100%', background: '#fff', borderRadius: '15px', boxShadow: '0 6px 20px rgba(0,0,0,0.08)', padding: '40px' }}>
        <h2 style={{ color: '#2c3e50', textAlign: 'center', marginBottom: '30px', fontSize: '2rem', fontWeight: 700 }}>Set New Password</h2>
        {error && !message && (
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
            <label style={{ display: 'block', color: '#3498db', fontWeight: 600, marginBottom: '10px' }}><FaLock style={{ marginRight: '10px' }} /> New Password</label>
            <input 
              type="password" 
              placeholder="Enter new password" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)} 
              style={{ width: '100%', padding: '12px', border: '1px solid #ddd', borderRadius: '6px', fontSize: '1rem', background: '#f9f9f9' }}
              required 
            />
          </div>
          <div>
            <label style={{ display: 'block', color: '#3498db', fontWeight: 600, marginBottom: '10px' }}><FaLock style={{ marginRight: '10px' }} /> Confirm Password</label>
            <input 
              type="password" 
              placeholder="Confirm new password" 
              value={confirmPassword} 
              onChange={(e) => setConfirmPassword(e.target.value)} 
              style={{ width: '100%', padding: '12px', border: '1px solid #ddd', borderRadius: '6px', fontSize: '1rem', background: '#f9f9f9' }}
              required 
            />
          </div>
          <button 
            type="submit" 
            disabled={loading}
            style={{ padding: '12px', background: loading ? '#bdc3c7' : 'linear-gradient(90deg, #2c3e50, #3498db)', border: 'none', borderRadius: '6px', color: 'white', fontWeight: 600, fontSize: '1.1rem', cursor: loading ? 'not-allowed' : 'pointer', transition: 'transform 0.2s' }}
          >
            {loading ? 'Resetting...' : 'Reset Password'}
          </button>
        </form>
        <p style={{ textAlign: 'center', marginTop: '20px', color: '#666', fontSize: '0.95rem' }}>
          Back to <Link to="/login" style={{ color: '#3498db', textDecoration: 'none', fontWeight: 500 }}>Login</Link>
        </p>
      </div>
    </div>
  );
};

export default ResetPassword;