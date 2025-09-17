import React, { useState } from 'react';
import { supabase } from '../supabase';
import './styles.css';

const LoginModal = ({ isOpen, onClose }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [captchaAnswer, setCaptchaAnswer] = useState('');
  const [captchaQuestion, setCaptchaQuestion] = useState('');
  const [captchaCorrectAnswer, setCaptchaCorrectAnswer] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const generateCaptcha = () => {
    const num1 = Math.floor(Math.random() * 10) + 1;
    const num2 = Math.floor(Math.random() * 10) + 1;
    const operators = ['+', '-', '*'];
    const operator = operators[Math.floor(Math.random() * operators.length)];
    const question = `${num1} ${operator} ${num2} = ?`;
    let answer;
    if (operator === '+') answer = num1 + num2;
    else if (operator === '-') answer = num1 - num2;
    else answer = num1 * num2;
    setCaptchaQuestion(question);
    setCaptchaCorrectAnswer(answer.toString());
  };

  React.useEffect(() => {
    if (isOpen) generateCaptcha();
  }, [isOpen]);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    if (!/[^@]+@[^@]+\.[^@]+/.test(email)) {
      setError('Invalid email format');
      setLoading(false);
      return;
    }
    if (!password) {
      setError('Password cannot be empty');
      setLoading(false);
      return;
    }
    if (captchaAnswer !== captchaCorrectAnswer) {
      setError('Incorrect CAPTCHA answer');
      setCaptchaAnswer('');
      generateCaptcha();
      setLoading(false);
      return;
    }

    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
    try {
      const { data, error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) throw error;
      navigate('/dashboard'); // Redirect to dashboard
    } catch (err) {
      setError('Login failed: ' + err.message);
    } finally {
      setLoading(false);
    }
      if (error) {
        setError('Invalid email or password');
        setLoading(false);
        return;
      }

      // Successful login, close modal or update app state
      onClose(); // Close the modal after successful login
    } catch (err) {
      setError('Login failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal">
        <h3>Welcome back!</h3>
        <form onSubmit={handleLogin}>
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <div className="security-section">
            <div className="security-title">🔢 Security Verification:</div>
            <div className="math-question">{captchaQuestion}</div>
            <input
              type="text"
              placeholder="Enter the answer"
              value={captchaAnswer}
              onChange={(e) => setCaptchaAnswer(e.target.value)}
              required
            />
          </div>
          {error && <p className="error">{error}</p>}
          <button type="submit" disabled={loading}>
            {loading ? 'Signing In...' : 'Sign In'}
          </button>
          <button type="button" onClick={onClose}>Close</button>
        </form>
        <button onClick={() => window.location.href = '/forgot-password'}>Forgot Password?</button>
      </div>
    </div>
  );
};

export default LoginModal;