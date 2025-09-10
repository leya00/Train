import React from 'react';
import { NavLink } from 'react-router-dom';

const NavBar: React.FC = () => {
  return (
    <nav style={navStyle}>
      <div style={logoStyle}>TrainAI</div>
      <div style={linkContainerStyle}>
        <NavLink to="/" end style={navLinkStyle}>
          Home
        </NavLink>
        <NavLink to="/researcher" style={navLinkStyle}>
          Researcher
        </NavLink>
        <NavLink to="/commuter" style={navLinkStyle}>
          Commuter
        </NavLink>
        <NavLink to="/schedual" style={navLinkStyle}>
          schedual
        </NavLink>
        <NavLink to="/statistics" style={navLinkStyle}>
          statistics
        </NavLink>
      </div>
    </nav>
  );
};

// Main navigation bar container for desktop only
const navStyle: React.CSSProperties = {
  display: 'flex',                  // Flex layout for horizontal alignment
  justifyContent: 'space-between', // Space between logo and links
  alignItems: 'center',             // Vertically center items
  padding: '2rem 3rem',             // Padding around nav
  backgroundColor: '#ffffff',          // background
  color: 'black',                   //  text
  position: 'fixed',                // Fixed at top of page
  width: '100%',                   // Full width
  top: 0,
  left: 0,
  zIndex: 1000,
  flexWrap: 'nowrap',   
    fontFamily: "'VT323', monospace",
           // No wrapping needed
};

// Logo styling
const logoStyle: React.CSSProperties = {
  fontWeight: 'bold',
  fontSize: '3rem',
  cursor: 'pointer',
  marginLeft: '4rem',           // Space on the left side
    fontFamily: "'VT323', monospace",

};

// Container for nav links
const linkContainerStyle: React.CSSProperties = {
  display: 'flex',                 // Horizontal layout for links
  gap: '1.5rem',                  // Spacing between links
  marginRight: '10rem',             // Space on the right side
  fontSize: '1.5rem',          // Font size for links
};

// Style for each nav link
const navLinkStyle = ({ isActive }: { isActive: boolean }): React.CSSProperties => ({
  color: isActive ? '#0a5fbf' : '#313233',  // Green highlight if active, else white
  textDecoration: 'none',
  fontWeight: isActive ? '700' : '500',
});


export default NavBar;
