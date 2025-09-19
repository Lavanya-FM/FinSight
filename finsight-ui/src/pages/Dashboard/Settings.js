// src/pages/Dashboard/Settings.js
import React, { useState, useEffect } from 'react';
import { FaBell } from 'react-icons/fa';  // Kept only non-SVG-conflicting icon

const Settings = ({ backendUrl }) => {
  const [settings, setSettings] = useState({ notifications: true, theme: 'light' });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchSettings();
  }, [backendUrl]);  // Added dependency for safety

  const fetchSettings = async () => {
    if (!backendUrl) {
      console.warn('No backendUrl provided; using defaults');
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${backendUrl}/api/settings`, {
        method: 'GET',
        credentials: 'include',
        mode: 'cors',  // Ensure CORS mode
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        const errorText = await response.text();  // Read as text to catch HTML errors (e.g., 404)
        console.error('Settings fetch failed:', response.status, errorText);
        throw new Error(`HTTP ${response.status}: ${errorText.substring(0, 100)}...`);
      }

      const data = await response.json();
      setSettings(data);
    } catch (err) {
      console.error('Error fetching settings:', err);
      setError(err.message);
      // Fallback to defaults (already set in state)
    } finally {
      setLoading(false);
    }
  };

  const handleSaveSettings = async () => {
    if (!backendUrl) {
      alert('No backend URL configured; cannot save settings.');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${backendUrl}/api/settings`, {
        method: 'PUT',
        credentials: 'include',
        mode: 'cors',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Settings save failed:', response.status, errorText);
        throw new Error(`HTTP ${response.status}: ${errorText.substring(0, 100)}...`);
      }

      alert('Settings saved successfully!');
      // Apply theme change immediately
      document.body.className = settings.theme === 'dark' ? 'dark-theme' : 'light-theme';
    } catch (err) {
      console.error('Error saving settings:', err);
      setError(err.message);
      alert(`Failed to save settings: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ maxWidth: 'min(600px, 90%)', margin: '0 auto', padding: '40px 20px', textAlign: 'center' }}>
        <p>Loading settings...</p>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 'min(600px, 90%)', margin: '0 auto', padding: '40px 20px', background: '#ffffff' }}>
      <h1 style={{ color: '#2c3e50', marginBottom: '30px', fontSize: '2rem', fontWeight: 700 }}>Settings</h1>
      {error && (
        <div style={{ background: '#fee', border: '1px solid #fcc', borderRadius: '6px', padding: '10px', marginBottom: '20px', color: '#c33' }}>
          <p>Error: {error}</p>
          <button onClick={fetchSettings} style={{ marginTop: '5px', padding: '4px 8px', background: '#fcc', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
            Retry
          </button>
        </div>
      )}
      <div style={{ background: '#f8f9fa', borderRadius: '10px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', padding: '30px' }}>
        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'flex', alignItems: 'center', color: '#3498db', fontWeight: 600, marginBottom: '10px' }}>
            <FaBell style={{ marginRight: '10px' }} />
            <span>Enable Notifications</span>
            <input
              type="checkbox"
              checked={settings.notifications}
              onChange={(e) => setSettings({ ...settings, notifications: e.target.checked })}
              style={{ marginLeft: '10px' }}
              disabled={loading}
            />
          </label>
        </div>
        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'flex', alignItems: 'center', color: '#3498db', fontWeight: 600, marginBottom: '10px' }}>
            <span>Theme</span>
            <select
              value={settings.theme}
              onChange={(e) => setSettings({ ...settings, theme: e.target.value })}
              style={{ marginLeft: '10px', padding: '8px', border: '1px solid #ddd', borderRadius: '6px', fontSize: '1rem' }}
              disabled={loading}
            >
              <option key="light" value="light">☀️ Light</option>
              <option key="dark" value="dark">🌙 Dark</option>
            </select>
          </label>
        </div>
        <button
          onClick={handleSaveSettings}
          disabled={loading}
          style={{ 
            padding: '12px 20px', 
            background: loading ? '#bdc3c7' : '#2c3e50', 
            color: 'white', 
            border: 'none', 
            borderRadius: '6px', 
            fontSize: '1rem', 
            cursor: loading ? 'not-allowed' : 'pointer' 
          }}
        >
          {loading ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
};

export default Settings;