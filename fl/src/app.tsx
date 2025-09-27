import { Routes, Route } from 'react-router-dom';
import NavBar from './components/Navbar';
import Home from './pages/Home';
import Researcher from './pages/Researcher';
import Commuter from './pages/Commuter';
import CommuterSchedule from './pages/commuter_schedule';
import ResearcherStatistics from './pages/researcher_statistics';

function App() {
  return (
    <>
      <NavBar />
      <div className="pt-20 px-4">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/researcher" element={<Researcher />} />
          <Route path="/researcher/statistics" element={<ResearcherStatistics />} />
          <Route path="/commuter" element={<Commuter />} />
          <Route path="/commuter/schedule" element={<CommuterSchedule />} />
        </Routes>
      </div>
    </>
  );
}

export default App;
