import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { FaEnvelope, FaLock, FaShieldAlt } from 'react-icons/fa';
import supabase from '../utils/supabaseClient';
import { useUser } from '../context/UserContext';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [captchaAnswer, setCaptchaAnswer] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { handleLogin } = useUser();

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

  if (!supabase) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', background: '#f8f9fa' }}>
        <div style={{ textAlign: 'center', padding: '2rem', background: 'white', borderRadius: '8px', boxShadow: '0 2px 10px rgba(0,0,0,0.1)' }}>
          <h2 style={{ color: '#e74c3c' }}>Configuration Error</h2>
          <p>Supabase URL or Key is missing in .env. Check console for details and restart the server.</p>
          <button onClick={() => window.location.reload()} style={{ padding: '10px 20px', background: '#3498db', color: 'white', border: 'none', borderRadius: '6px' }}>Reload</button>
        </div>
      </div>
    );
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    if (!email.trim().match(/[^@]+@[^@]+\.[^@]+/)) {
      setError('Invalid email format');
      setLoading(false);
      return;
    }
    if (!password.trim()) {
      setError('Password cannot be empty');
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
      console.log('Attempting login for:', email);

      const { data: { session }, error: authError } = await supabase.auth.signInWithPassword({
        email: email.trim(),
        password: password.trim(),
      });

      if (authError) {
        throw new Error(authError.message || 'Login failed. Please try again.');
      }

      if (!session || !session.user) {
        throw new Error('Login failed: No session created');
      }

      const { data: userData, error: fetchError } = await supabase
        .from('users')
        .select('user_id, role, username, email')
        .eq('user_id', session.user.id)
        .single();

      if (fetchError || !userData) {
        throw new Error('Failed to fetch user profile');
      }

      const { error: updateError } = await supabase
        .from('users')
        .update({ last_login: new Date().toISOString() })
        .eq('user_id', userData.user_id);

      if (updateError) {
        console.warn('Failed to update last_login:', updateError);
      }

      document.cookie = `supabase_token=${session.access_token}; path=/; max-age=${session.expires_in}; secure=false; samesite=lax`;

      console.log(`User ${email} logged in successfully with user_id: ${userData.user_id} and role: ${userData.role}`);
      handleLogin(userData.role, userData.user_id, userData.username, userData.email);

      navigate('/dashboard');

    } catch (error) {
      console.error('Login Error:', error);
      setError(error.message || 'Login failed. Please check your credentials.');
      setCaptcha(generateCaptcha());
      setCaptchaAnswer('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh', minWidth: '80vh', background: 'linear-gradient(135deg, #e6f0fa 0%, #f8f9fa 100%)', padding: '20px 10px' }}>
      <div style={{ maxWidth: '1000px', width: '100%', background: '#fff', borderRadius: '15px', boxShadow: '0 6px 20px rgba(0,0,0,0.08)', padding: '40px' }}>
        <h2 style={{ color: '#2c3e50', textAlign: 'center', marginBottom: '30px', fontSize: '2rem', fontWeight: 700 }}>Login to FinSight</h2>
        {error && (
          <div style={{ color: '#e74c3c', textAlign: 'center', marginBottom: '20px', fontSize: '0.95rem', background: '#fee', padding: '15px', borderRadius: '6px', border: '1px solid #fcc' }}>
            {error}
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
          <button 
            type="submit" 
            disabled={loading}
            style={{ padding: '12px', background: loading ? '#bdc3c7' : 'linear-gradient(90deg, #2c3e50, #3498db)', border: 'none', borderRadius: '6px', color: 'white', fontWeight: 600, fontSize: '1.1rem', cursor: loading ? 'not-allowed' : 'pointer', transition: 'transform 0.2s' }}
          >
            {loading ? 'Signing In...' : 'Sign In'}
          </button>
        </form>
        <Link 
          to="/forgot-password"
          style={{ display: 'block', textAlign: 'center', marginTop: '20px', color: '#3498db', textDecoration: 'none', fontSize: '1rem', fontWeight: 500 }}
        >
          Forgot Password?
        </Link>
        <p style={{ textAlign: 'center', marginTop: '20px', color: '#666', fontSize: '0.95rem' }}>
          Don't have an account? <Link to="/register" style={{ color: '#3498db', textDecoration: 'none', fontWeight: 500 }}>Sign Up</Link>
        </p>
      </div>
    </div>
  );
};

export default Login;