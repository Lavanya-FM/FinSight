import React from 'react';
import { 
  BrowserRouter as Router, 
  Route, 
  Routes, 
  Navigate 
} from 'react-router-dom';

import Header from './components/Header';
import Footer from './components/Footer';
import Home from './pages/Home';
import Features from './pages/Features';
import ApiAccess from './pages/ApiAccess';
import Pricing from './pages/Pricing';
import Contact from './pages/Contact';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import ForgotPassword from './pages/ForgotPassword';
import ResetPassword from './pages/ResetPassword';
import { UserProvider, useUser } from './context/UserContext';

function AppContent() {
  const { isLoggedIn, userRole, user, loading, handleLogout } = useUser();

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        background: '#ffffff'
      }}>
        <div style={{
          textAlign: 'center',
          padding: '2rem',
          background: '#f8f9fa',
          borderRadius: '10px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}>
          <div style={{
            border: '4px solid #f3f3f3',
            borderTop: '4px solid #2c3e50',
            borderRadius: '50%',
            width: '50px',
            height: '50px',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 1rem'
          }}></div>
          <p style={{ color: '#2c3e50' }}>Loading FinSight...</p>
          <style>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
        </div>
      </div>
    );
  }

  const onLogout = () => {
    handleLogout(() => {
      window.location.href = '/login'; // Redirect after logout
    });
  };

  return (
    <div className="app-container">
      <Header 
        isLoggedIn={isLoggedIn} 
        userRole={userRole} 
        onLogout={onLogout}
      />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/features" element={<Features />} />
          <Route path="/api-access" element={<ApiAccess />} />
          <Route path="/pricing" element={<Pricing />} />
          <Route path="/contact" element={<Contact />} />
          <Route 
            path="/login" 
            element={!isLoggedIn ? <Login /> : <Navigate to="/dashboard" />}
          />
          <Route 
            path="/register" 
            element={!isLoggedIn ? <Register /> : <Navigate to="/dashboard" />}
          />
          <Route 
            path="/forgot-password" 
            element={!isLoggedIn ? <ForgotPassword /> : <Navigate to="/dashboard" />}
          />
          <Route 
            path="/reset-password" 
            element={!isLoggedIn ? <ResetPassword /> : <Navigate to="/dashboard" />}
          />
          <Route 
            path="/dashboard/*" 
            element={isLoggedIn ? <Dashboard onLogout={onLogout} /> : <Navigate to="/login" />}
          />
        </Routes>
      </main>
      {!isLoggedIn && <Footer />}
    </div>
  );
}

export default function AppWrapper() {
  return (
    <UserProvider>
      <Router future={{ 
        v7_startTransition: true, 
        v7_relativeSplatPath: true 
      }}>
        <AppContent />
      </Router>
    </UserProvider>
  );
}