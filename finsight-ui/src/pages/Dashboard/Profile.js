import React, { useState, useEffect } from 'react';
import { FaEdit, FaSave } from 'react-icons/fa';
import { supabase } from '../../supabase';

const Profile = () => {
  const [profile, setProfile] = useState({ username: '', email: '', role: '' });
  const [editing, setEditing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchProfile();
  }, []);

  const fetchProfile = async () => {
    try {
      setLoading(true);
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error('No authenticated user found');

      const { data, error } = await supabase
        .from('users')
        .select('username, email, role')
        .eq('user_id', user.id)
        .single();
      if (error) throw new Error(error.message);
      setProfile(data);
    } catch (err) {
      setError('Error fetching profile: ' + err.message);
      console.error('Profile fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    try {
      setLoading(true);
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error('No authenticated user found');

      const { error } = await supabase
        .from('users')
        .update({ username: profile.username, email: profile.email })
        .eq('user_id', user.id);
      if (error) throw new Error(error.message);
      setEditing(false);
      alert('Profile updated successfully');
    } catch (err) {
      setError('Error updating profile: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div style={{ textAlign: 'center', padding: '40px', color: '#2c3e50' }}>Loading profile...</div>;
  if (error) return <div style={{ textAlign: 'center', padding: '40px', color: '#e74c3c' }}>Error: {error}</div>;

  return (
    <div style={{ maxWidth: 'min(600px, 90%)', margin: '0 auto', padding: '40px 20px', background: '#ffffff' }}>
      <h1 style={{ color: '#2c3e50', marginBottom: '30px', fontSize: '2rem', fontWeight: 700 }}>Profile - {profile.username || 'User'}</h1>
      <div style={{ background: '#f8f9fa', borderRadius: '10px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', padding: '30px' }}>
        <div style={{ display: 'grid', gap: '20px' }}>
          <div>
            <label style={{ display: 'block', color: '#3498db', fontWeight: 600, marginBottom: '10px' }}>Username</label>
            <input
              value={profile.username}
              onChange={(e) => setProfile({ ...profile, username: e.target.value })}
              disabled={!editing}
              style={{ width: '100%', padding: '12px', border: '1px solid #ddd', borderRadius: '6px', fontSize: '1rem', background: editing ? '#fff' : '#f1f1f1' }}
            />
          </div>
          <div>
            <label style={{ display: 'block', color: '#3498db', fontWeight: 600, marginBottom: '10px' }}>Email</label>
            <input
              value={profile.email}
              onChange={(e) => setProfile({ ...profile, email: e.target.value })}
              disabled={!editing}
              style={{ width: '100%', padding: '12px', border: '1px solid #ddd', borderRadius: '6px', fontSize: '1rem', background: editing ? '#fff' : '#f1f1f1' }}
            />
          </div>
          <div>
            <label style={{ display: 'block', color: '#3498db', fontWeight: 600, marginBottom: '10px' }}>Role</label>
            <input
              value={profile.role}
              disabled
              style={{ width: '100%', padding: '12px', border: '1px solid #ddd', borderRadius: '6px', background: '#f1f1f1', fontSize: '1rem' }}
            />
          </div>
        </div>
        <div style={{ marginTop: '30px', display: 'flex', gap: '15px', justifyContent: 'flex-end' }}>
          {!editing ? (
            <button onClick={() => setEditing(true)} style={{ padding: '12px 20px', background: '#2c3e50', color: 'white', border: 'none', borderRadius: '6px', display: 'flex', alignItems: 'center', gap: '5px', fontSize: '1rem' }}>
              <FaEdit /> Edit
            </button>
          ) : (
            <>
              <button onClick={handleSave} style={{ padding: '12px 20px', background: '#2ecc71', color: 'white', border: 'none', borderRadius: '6px', display: 'flex', alignItems: 'center', gap: '5px', fontSize: '1rem' }}>
                <FaSave /> Save
              </button>
              <button onClick={() => setEditing(false)} style={{ padding: '12px 20px', background: '#bdc3c7', color: 'white', border: 'none', borderRadius: '6px', fontSize: '1rem' }}>
                Cancel
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default Profile;