// New file: src/components/Sidebar.js
import React from 'react';
import { Link } from 'react-router-dom';

const Sidebar = ({ items, activeId, user }) => {
  return (
    <div style={{
      width: '250px',
      background: 'linear-gradient(135deg, #2c3e50 0%, #3498db 100%)',
      color: 'white',
      padding: '20px 0',
      boxShadow: '2px 0 10px rgba(0,0,0,0.1)'
    }}>
      {/* User Info */}
      {user && (
        <div style={{ padding: '0 20px 20px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <h3 style={{ margin: 0, fontSize: '1.2rem' }}>{user.username}</h3>
          <p style={{ margin: '5px 0 0', opacity: 0.8, fontSize: '0.9rem' }}>{user.role}</p>
        </div>
      )}

      {/* Menu Items */}
      <nav style={{ padding: '20px 0' }}>
        {items.map((item) => (
          item.id === 'logout' ? (
            <button
              key={item.id}
              onClick={item.onClick}
              style={{
                width: '100%',
                padding: '12px 20px',
                background: 'transparent',
                border: 'none',
                color: 'white',
                textAlign: 'left',
                cursor: 'pointer',
                fontSize: '1rem',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                transition: 'background 0.3s'
              }}
              onMouseEnter={(e) => e.target.style.background = 'rgba(255,255,255,0.1)'}
              onMouseLeave={(e) => e.target.style.background = 'transparent'}
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ) : (
            <Link
              key={item.id}
              to={`/dashboard/${item.id}`}
              style={{
                display: 'block',
                padding: '12px 20px',
                color: activeId === item.id ? 'white' : 'rgba(255,255,255,0.8)',
                textDecoration: 'none',
                fontSize: '1rem',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                transition: 'all 0.3s',
                background: activeId === item.id ? 'rgba(255,255,255,0.1)' : 'transparent'
              }}
              onMouseEnter={(e) => {
                if (activeId !== item.id) e.target.style.background = 'rgba(255,255,255,0.1)';
              }}
              onMouseLeave={(e) => {
                if (activeId !== item.id) e.target.style.background = 'transparent';
              }}
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
            </Link>
          )
        ))}
      </nav>
    </div>
  );
};

export default Sidebar;