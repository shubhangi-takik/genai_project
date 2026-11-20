import React from 'react';

const Navbar = ({ title }) => {
  return (
    <header className="navbar">
      <h1>{title}</h1>
      <div className="navbar-actions">
        {/* Placeholder for future features like user profile, theme toggle */}
      </div>
    </header>
  );
};

export default Navbar;
