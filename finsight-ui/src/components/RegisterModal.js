import React, { useState, useEffect } from 'react';
import { supabase } from '../supabase';
import './styles.css';

const RegisterModal = ({ isOpen, onClose }) => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('user');
  const [captchaAnswer, setCaptchaAnswer] = useState('');
  const [captchaQuestion, setCaptchaQuestion] = useState('');
  const [captchaCorrectAnswer, setCaptchaCorrectAnswer] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [isRegistered, setIsRegistered] = useState(false);

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

  useEffect(() => {
    if (isOpen) generateCaptcha();
  }, [isOpen]);

  const handleRegister = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    if (!username) {
      setError('Name cannot be empty');
      setLoading(false);
      return;
    }
    if (!/[^@]+@[^@]+\.[^@]+/.test(email)) {
      setError('Invalid email format');
      setLoading(false);
      return;
    }
    if (password.length < 8 || !/[!@#$%^&*()_+]/.test(password)) {
      setError('Password must be at least 8 characters and include a special character');
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
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
      });

      if (error) {
        if (error.message.includes('duplicate key')) {
          setError('Email already registered');
        } else {
          setError('Registration failed: ' + error.message);
        }
        setLoading(false);
        return;
      }

      const user = data.user;
      // Insert user data into the users table
      const { error: dbError } = await supabase.from('users').insert([
        {
          user_id: user.id,
          username,
          email,
          role,
          created_at: new Date().toISOString(),
        },
      ]);

      if (dbError) {
        setError('Failed to save user data: ' + dbError.message);
        setLoading(false);
        return;
      }

      // Set registered state and clear form
      setIsRegistered(true);
      setUsername('');
      setEmail('');
      setPassword('');
      setRole('user');
      setCaptchaAnswer('');
      generateCaptcha();
    } catch (err) {
      setError('Registration failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal">
        <h3>{isRegistered ? 'Registration Successful' : 'Create Account'}</h3>
        {isRegistered ? (
          <div className="success-message">
            <p>Registered successfully at {new Date().toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })}!</p>
            <button onClick={onClose}>Close</button>
          </div>
        ) : (
          <form onSubmit={handleRegister}>
            <input
              type="text"
              placeholder="Enter your name"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
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
            <div className="role-selection">
              <label>
                <input
                  type="radio"
                  value="user"
                  checked={role === 'user'}
                  onChange={() => setRole('user')}
                />
                👤 User
              </label>
              <label>
                <input
                  type="radio"
                  value="admin"
                  checked={role === 'admin'}
                  onChange={() => setRole('admin')}
                />
                👑 Admin
              </label>
            </div>
            <div className="security-section">
              <div className="security-title">🔢 Security Verification:</div>
              <div className="math-question">{captchaQuestion}</div>
              <input
                type="text"
                placeholder="Enter answer"
                value={captchaAnswer}
                onChange={(e) => setCaptchaAnswer(e.target.value)}
                required
              />
            </div>
            {error && <p className="error">{error}</p>}
            <button type="submit" disabled={loading}>
              {loading ? 'Creating Account...' : 'Create Account'}
            </button>
            <button type="button" onClick={onClose}>Close</button>
          </form>
        )}
      </div>
    </div>
  );
};

export default RegisterModal;