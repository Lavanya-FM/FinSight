// src/pages/Register.js
import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import supabase from '../utils/supabaseClient'; // Shared client to avoid multiple instances
import { FaUser, FaEnvelope, FaLock, FaShieldAlt, FaCrown } from 'react-icons/fa';

const Register = ({ onRegister }) => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('user');
  const [captchaAnswer, setCaptchaAnswer] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  // Generate CAPTCHA
  function generateCaptcha() {
    const num1 = Math.floor(Math.random() * 10) + 1;
    const num2 = Math.floor(Math.random() * 10) + 1;
    const operators = ['+', '-', '*'];
    const operator = operators[Math.floor(Math.random() * operators.length)];
    let answer;
    if (operator === '+') answer = num1 + num2;
    else if (operator === '-') answer = num1 - num2;
    else answer = num1 * num2;
    return { question: `${num1} ${operator} ${num2} = ?`, answer: answer.toString() };
  }
  const [captcha, setCaptcha] = useState(generateCaptcha());

  // Show config error if Supabase not initialized
  if (!supabase) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', background: '#f8f9fa' }}>
        <div style={{ textAlign: 'center', padding: '2rem', background: 'white', borderRadius: '8px', boxShadow: '0 2px 10px rgba(0,0,0,0.1)' }}>
          <h2 style={{ color: '#e74c3c' }}>Configuration Error</h2>
          <p>Supabase URL or Key is missing in .env. Check console for details and restart the server.</p>
          <button onClick={() => window.location.reload()} style={{ padding: '0.5rem 1rem', marginTop: '1rem' }}>Reload</button>
        </div>
      </div>
    );
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    // Validation with trim
    const trimmedName = name.trim();
    const trimmedEmail = email.trim();
    if (!trimmedName) {
      setError('Name cannot be empty');
      setLoading(false);
      return;
    }
    if (!/[^@]+@[^@]+\.[^@]+/.test(trimmedEmail)) {
      setError('Invalid email format');
      setLoading(false);
      return;
    }
    if (password.length < 8 || !/[!@#$%^&*()_+]/.test(password)) {
      setError('Password must be at least 8 characters and include a special character');
      setLoading(false);
      return;
    }
    if (captchaAnswer !== captcha.answer) {
      setError('Incorrect CAPTCHA answer');
      setCaptcha(generateCaptcha());
      setCaptchaAnswer('');
      setLoading(false);
      return;
    }

    try {
      console.log('Attempting signup for:', trimmedEmail);

      // Sign up without email confirmation
      const { data: signUpData, error: authError } = await supabase.auth.signUp({
        email: trimmedEmail,
        password,
        options: {
          data: {
            username: trimmedName, // Pass to trigger
            role: role // Pass to trigger
          }
        }
      });

      console.log('Signup response:', { data: signUpData, error: authError });

      if (authError) {
        console.error('Auth Error Details:', authError); // Log full error
        if (authError.message.includes('Database error saving new user')) {
          throw new Error('Database error during signup (likely trigger/RLS issue). Check Supabase dashboard logs for details. Error: ' + (authError.details || authError.message));
        }
        if (authError.message.includes('already registered') || authError.code === 'duplicate_user') {
          throw new Error('Email already registered. Please log in instead.');
        }
        throw new Error(authError.message || 'Signup failed. Please try again.');
      }

      if (!signUpData.user) {
        throw new Error('Signup failed: No user created. Check Supabase logs.');
      }

      // Automatically sign in the user after signup
      const { data: sessionData, error: signInError } = await supabase.auth.signInWithPassword({
        email: trimmedEmail,
        password
      });

      if (signInError) {
        throw new Error('Auto-login failed: ' + signInError.message);
      }

      if (!sessionData.session || !sessionData.user) {
        throw new Error('Auto-login failed: No session created');
      }

      // Set supabase_token cookie (critical for FastAPI session validation)
      document.cookie = `supabase_token=${sessionData.session.access_token}; path=/; max-age=${sessionData.session.expires_in}; secure=false; samesite=lax`;

      console.log(`User ${trimmedName} registered and logged in successfully with user_id: ${sessionData.user.id}`);
      onRegister?.(); // Optional callback
      navigate('/dashboard'); // Redirect to dashboard instead of login

    } catch (err) {
      console.error('Registration Error:', err);
      setError(err.message || 'Registration failed. Please contact support if the issue persists.');
      setCaptcha(generateCaptcha());
      setCaptchaAnswer('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minWidth: '95vh', background: 'linear-gradient(135deg, #e6f0fa 0%, #f8f9fa 100%)', padding: '40px 20px' }}>
      <div style={{ maxWidth: 'min(600px, 90%)', width: '100%', background: '#fff', borderRadius: '15px', boxShadow: '0 6px 20px rgba(0,0,0,0.08)', padding: '40px' }}>
        <h2 style={{ color: '#2c3e50', textAlign: 'center', marginBottom: '30px', fontSize: '2rem', fontWeight: 700 }}>Create Account</h2>
        {error && (
          <div style={{ color: '#e74c3c', textAlign: 'center', marginBottom: '20px', fontSize: '0.95rem', background: '#fee', padding: '15px', borderRadius: '6px', border: '1px solid #fcc' }}>
            {error}
          </div>
        )}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div>
            <label style={{ display: 'block', color: '#3498db', fontWeight: 600, marginBottom: '10px' }}><FaUser style={{ marginRight: '10px' }} /> Full Name</label>
            <input 
              type="text" 
              placeholder="Enter your full name" 
              value={name} 
              onChange={(e) => setName(e.target.value)} 
              style={{ width: '100%', padding: '12px', border: '1px solid #ddd', borderRadius: '6px', fontSize: '1rem', background: '#f9f9f9' }}
              required 
            />
          </div>
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
          <div>
            <label style={{ display: 'block', color: '#3498db', fontWeight: 600, marginBottom: '10px' }}><FaLock style={{ marginRight: '10px' }} /> Password</label>
            <input 
              type="password" 
              placeholder="Enter your password" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)} 
              style={{ width: '100%', padding: '12px', border: '1px solid #ddd', borderRadius: '6px', fontSize: '1rem', background: '#f9f9f9' }}
              required 
            />
          </div>
          <div>
            <label style={{ display: 'block', color: '#3498db', fontWeight: 600, marginBottom: '10px' }}><FaShieldAlt style={{ marginRight: '10px' }} /> Security Verification</label>
            <p style={{ color: '#666', marginBottom: '10px' }}>{captcha.question}</p>
            <input 
              type="text" 
              placeholder="Enter the answer" 
              value={captchaAnswer} 
              onChange={(e) => setCaptchaAnswer(e.target.value)} 
              style={{ width: '100%', padding: '12px', border: '1px solid #ddd', borderRadius: '6px', fontSize: '1rem', background: '#f9f9f9' }}
              required 
            />
          </div>
          <div>
            <label style={{ display: 'block', color: '#3498db', fontWeight: 600, marginBottom: '10px' }}>Choose your role:</label>
            <div style={{ display: 'flex', gap: '20px', justifyContent: 'center' }}>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  type="radio"
                  value="user"
                  checked={role === 'user'}
                  onChange={(e) => setRole(e.target.value)}
                  style={{ marginRight: '10px' }}
                />
                <FaUser style={{ marginRight: '5px' }} /> User
              </label>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  type="radio"
                  value="admin"
                  checked={role === 'admin'}
                  onChange={(e) => setRole(e.target.value)}
                  style={{ marginRight: '10px' }}
                />
                <FaCrown style={{ marginRight: '5px' }} /> Admin
              </label>
            </div>
          </div>
          <button 
            type="submit" 
            disabled={loading}
            style={{ padding: '12px', background: loading ? '#bdc3c7' : 'linear-gradient(90deg, #2c3e50, #3498db)', border: 'none', borderRadius: '6px', color: 'white', fontWeight: 600, fontSize: '1.1rem', cursor: loading ? 'not-allowed' : 'pointer', transition: 'transform 0.2s' }}
          >
            {loading ? 'Creating Account...' : 'Create Account'}
          </button>
        </form>
        <p style={{ textAlign: 'center', marginTop: '20px', color: '#666', fontSize: '0.95rem' }}>
          Already have an account? <Link to="/login" style={{ color: '#3498db', textDecoration: 'none', fontWeight: 500 }}>Sign In</Link>
        </p>
      </div>
    </div>
  );
};

export default Register;