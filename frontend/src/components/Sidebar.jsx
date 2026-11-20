import React from 'react';
import { NavLink } from 'react-router-dom';
import { Home, Image, Layers, Cpu, AlignLeft, Sparkles, BarChart2 } from 'lucide-react';

const Sidebar = () => {
  const navItems = [
    { path: '/', label: 'Home', icon: Home },
    { path: '/autoencoder', label: 'Autoencoder', icon: Layers },
    { path: '/vae', label: 'VAE', icon: Layers },
    { path: '/gan', label: 'GAN', icon: Image },
    { path: '/transformer', label: 'Transformer', icon: AlignLeft },
    { path: '/diffusion', label: 'Diffusion', icon: Sparkles },
    { path: '/comparison', label: 'Model Comparison', icon: BarChart2 },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <Cpu className="logo-icon" />
        <h2>GenAI Design</h2>
      </div>
      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => (isActive ? 'nav-item active' : 'nav-item')}
            >
              <Icon size={20} />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>
    </aside>
  );
};

export default Sidebar;
