// src/components/Header.js
import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Header = ({ isLoggedIn, userRole, onLogout }) => {
  const location = useLocation();

  const handleNavClick = (e, sectionId, routePath) => {
    e.preventDefault();
    if (location.pathname === '/' && sectionId) {
      const section = document.getElementById(sectionId);
      if (section) {
        section.scrollIntoView({ behavior: 'smooth' });
      }
    } else {
      window.location.href = routePath || `/#${sectionId}`;
    }
  };

  const isDashboard = location.pathname.startsWith('/dashboard');
  const navItems = isDashboard ? [] : [
    { href: '/features', label: 'Features' },
    { href: '/api-access', label: 'API Access' },
    { href: '/pricing', label: 'Pricing' },
    { href: '/contact', label: 'Contact' }
  ];

  return (
    <header className="navbar" style={isDashboard ? { position: 'relative' } : {}}>
      <div className="header-content">
        <div className="header-logo">Q</div>
        <div>
          <h1 className="header-title">FinSight</h1>
          <p>AI Financial Insights</p>
        </div>
      </div>
      {!isDashboard && (
        <nav>
          <ul className="nav-links">
            {navItems.map((item) => (
              <li key={item.href}>
                <a href={item.href} onClick={(e) => handleNavClick(e, null, item.href)}>
                  {item.label}
                </a>
              </li>
            ))}
            {isLoggedIn ? (
              <li><Link to="/dashboard">Dashboard</Link></li>
            ) : (
              <>
                <li><Link to="/login">Login</Link></li>
                <li><Link to="/register">Register</Link></li>
              </>
            )}
          </ul>
        </nav>
      )}
      {isDashboard && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span>Welcome, {userRole}</span>
          <button onClick={onLogout} style={{ background: 'none', border: '1px solid white', color: 'white', padding: '5px 10px', borderRadius: '4px' }}>Logout</button>
        </div>
      )}
    </header>
  );
};

export default Header;