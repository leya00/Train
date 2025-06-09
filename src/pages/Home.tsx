import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-4 p-4">
      <h1 className="text-4xl font-bold">Train Detection Interface</h1>
      <p className="text-lg">Choose your view</p>
      <div className="flex gap-4">
        <Link to="/researcher" className="bg-blue-500 text-white px-4 py-2 rounded-xl">Researcher View</Link>
        <Link to="/commuter" className="bg-green-500 text-white px-4 py-2 rounded-xl">Commuter View</Link>
      </div>
    </div>
  );
}