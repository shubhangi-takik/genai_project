import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Autoencoder from './pages/Autoencoder';
import VAE from './pages/VAE';
import GAN from './pages/GAN';
import Transformer from './pages/Transformer';
import Diffusion from './pages/Diffusion';
import Comparison from './pages/Comparison';
import { AutoencoderResultProvider } from './context/AutoencoderResultContext';

const Layout = ({ children }) => {
  const location = useLocation();
  
  // Map routes to titles
  const getTitle = () => {
    switch (location.pathname) {
      case '/': return 'Home';
      case '/autoencoder': return 'Autoencoder';
      case '/vae': return 'VAE';
      case '/gan': return 'GAN';
      case '/transformer': return 'Transformer';
      case '/diffusion': return 'Diffusion Model';
      case '/comparison': return 'Model Comparison';
      default: return 'GenAI Design';
    }
  };

  return (
    <div className="app-container">
      <Sidebar />
      <main className="main-content">
        <Navbar title={getTitle()} />
        {children}
      </main>
    </div>
  );
};

const App = () => {
  return (
    <AutoencoderResultProvider>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/autoencoder" element={<Autoencoder />} />
            <Route path="/vae" element={<VAE />} />
            <Route path="/gan" element={<GAN />} />
            <Route path="/transformer" element={<Transformer />} />
            <Route path="/diffusion" element={<Diffusion />} />
            <Route path="/comparison" element={<Comparison />} />
          </Routes>
        </Layout>
      </Router>
    </AutoencoderResultProvider>
  );
};

export default App;
