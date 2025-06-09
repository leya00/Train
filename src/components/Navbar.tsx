import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';

const NavBar: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth <= 768);
      if (window.innerWidth > 768) setIsOpen(false); // reset on desktop
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <nav style={navStyle}>
      <div style={logoStyle}>TrainAI</div>

      {isMobile && (
        <div style={hamburgerStyle} onClick={() => setIsOpen(!isOpen)}>
          <div style={lineStyle}></div>
          <div style={lineStyle}></div>
          <div style={lineStyle}></div>
        </div>
      )}

      <div
        style={{
          ...linkContainerStyle,
          ...(isMobile ? (isOpen ? mobileMenuOpenStyle : mobileMenuClosedStyle) : {}),
        }}
      >
        <NavLink to="/" end style={navLinkStyle} onClick={() => setIsOpen(false)}>
          Home
        </NavLink>
        <NavLink to="/researcher" style={navLinkStyle} onClick={() => setIsOpen(false)}>
          Researcher
        </NavLink>
        <NavLink to="/commuter" style={navLinkStyle} onClick={() => setIsOpen(false)}>
          Commuter
        </NavLink>
      </div>
    </nav>
  );
};

// Same styles as before...
const navStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: '1rem 2rem',
  backgroundColor: '#222',
  color: '#fff',
  position: 'fixed',
  width: '100%',
  top: 0,
  left: 0,
  zIndex: 1000,
  flexWrap: 'wrap',
};

const logoStyle: React.CSSProperties = {
  fontWeight: 'bold',
  fontSize: '1.5rem',
  cursor: 'pointer',
};

const hamburgerStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  cursor: 'pointer',
  gap: '5px',
};

const lineStyle: React.CSSProperties = {
  width: '25px',
  height: '3px',
  backgroundColor: '#fff',
  borderRadius: '2px',
};

const linkContainerStyle: React.CSSProperties = {
  display: 'flex',
  gap: '1.5rem',
};

const mobileMenuOpenStyle: React.CSSProperties = {
  flexDirection: 'column',
  width: '100%',
  marginTop: '1rem',
};

const mobileMenuClosedStyle: React.CSSProperties = {
  display: 'none',
};

const navLinkStyle = ({ isActive }: { isActive: boolean }): React.CSSProperties => ({
  color: isActive ? '#1DB954' : '#fff',
  textDecoration: 'none',
  fontWeight: isActive ? '700' : '500',
});

export default NavBar;
