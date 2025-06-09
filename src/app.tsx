import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Researcher from './pages/Researcher';
import Commuter from './pages/Commuter';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/researcher" element={<Researcher />} />
        <Route path="/commuter" element={<Commuter />} />
      </Routes>
    </BrowserRouter>
  );
}