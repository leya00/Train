import { Routes, Route } from 'react-router-dom';
import NavBar from './components/Navbar';
import Home from './pages/Home';
import Researcher from './pages/Researcher';
import Commuter from './pages/Commuter';

function App() {
  return (
    <>
      <NavBar />
      <div className="pt-20 px-4">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/researcher" element={<Researcher />} />
          <Route path="/commuter" element={<Commuter />} />
        </Routes>
       </div>
    </>
  );
}

export default App;
