// src/pages/Dashboard/Settings.js
import React, { useState, useEffect } from 'react';
import { FaBell, FaMoon, FaSun } from 'react-icons/fa';

const Settings = ({ backendUrl }) => {
  const [settings, setSettings] = useState({ notifications: true, theme: 'light' });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${backendUrl}/api/settings`, {
        method: 'GET',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        const data = await response.json();
        setSettings(data);
      }
    } catch (err) {
      console.error('Error fetching settings:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveSettings = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${backendUrl}/api/settings`, {
        method: 'PUT',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
      if (response.ok) {
        alert('Settings saved successfully');
        // Apply theme change
        document.body.className = settings.theme === 'dark' ? 'dark-theme' : 'light-theme';
      } else {
        const errorText = await response.text();
        console.error('Settings save failed:', errorText);
        alert('Failed to save settings');
      }
    } catch (err) {
      console.error('Error saving settings:', err);
      alert('Error saving settings');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 'min(600px, 90%)', margin: '0 auto', padding: '40px 20px', background: '#ffffff' }}>
      <h1 style={{ color: '#2c3e50', marginBottom: '30px', fontSize: '2rem', fontWeight: 700 }}>Settings</h1>
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
              <option value="light"><FaSun style={{ marginRight: '5px' }} /> Light</option>
              <option value="dark"><FaMoon style={{ marginRight: '5px' }} /> Dark</option>
            </select>
          </label>
        </div>
        <button
          onClick={handleSaveSettings}
          disabled={loading}
          style={{ padding: '12px 20px', background: '#2c3e50', color: 'white', border: 'none', borderRadius: '6px', fontSize: '1rem', cursor: loading ? 'not-allowed' : 'pointer' }}
        >
          {loading ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
};

export default Settings;