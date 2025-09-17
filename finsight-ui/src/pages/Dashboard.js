// src/pages/Dashboard.js
import React, { useState, useEffect } from 'react';
import { useLocation, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import LoanAnalyticsSystem from './Dashboard/LoanAnalyticsSystem';
import Profile from './Dashboard/Profile';
import Reports from './Dashboard/Reports';
import Settings from './Dashboard/Settings';
import HelpAndSupport from './Dashboard/HelpAndSupport';
import { useUser } from '../context/UserContext';
import Header from '../components/Header';

const Dashboard = ({ onLogout }) => {
  const { user, backendUrl, isLoggedIn } = useUser();
  const location = useLocation();
  const [activeSection, setActiveSection] = useState('dashboard');

  useEffect(() => {
    if (!isLoggedIn) {
      console.log('User not logged in, redirecting to login');
      return;
    }
    if (location.pathname.includes('profile')) setActiveSection('profile');
    else if (location.pathname.includes('reports')) setActiveSection('reports');
    else if (location.pathname.includes('settings')) setActiveSection('settings');
    else if (location.pathname.includes('help')) setActiveSection('help');
    else if (location.pathname.includes('upload')) setActiveSection('upload');
    else setActiveSection('dashboard');
  }, [location.pathname, isLoggedIn]);

  if (!isLoggedIn) return <Navigate to="/login" />;

  const sidebarItems = [
    { id: 'dashboard', label: 'Dashboard', icon: '🏠' }, // Added Dashboard entry
    { id: 'upload', label: 'Upload Documents', icon: '📄' },
    { id: 'profile', label: 'Profile', icon: '👤' },
    { id: 'reports', label: 'Reports', icon: '📊' },
    { id: 'settings', label: 'Settings', icon: '⚙️' },
    { id: 'help', label: 'Help & Support', icon: '❓' },
    { id: 'logout', label: 'Logout', icon: '🚪', onClick: onLogout }
  ];

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: '#ffffff' }}>
      <Sidebar items={sidebarItems} activeId={activeSection} user={user} />
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <Header isLoggedIn={isLoggedIn} userRole={user?.role} onLogout={onLogout} style={{ position: 'relative' }} />
        <div style={{ flex: 1, padding: '40px 20px', overflowY: 'auto' }}>
          <Routes>
            <Route path="/" element={<div style={{ padding: '20px', background: '#f8f9fa', borderRadius: '10px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}><h2 style={{ color: '#2c3e50' }}>Welcome to Your Dashboard</h2><p style={{ color: '#666' }}>Manage your financial insights here.</p></div>} />
            <Route path="/upload" element={<LoanAnalyticsSystem backendUrl={backendUrl} />} />
            <Route path="/profile" element={<Profile backendUrl={backendUrl} />} />
            <Route path="/reports" element={<Reports backendUrl={backendUrl} />} />
            <Route path="/settings" element={<Settings backendUrl={backendUrl} />} />
            <Route path="/help" element={<HelpAndSupport />} />
            <Route path="*" element={<Navigate to="/dashboard" />} />
          </Routes>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;