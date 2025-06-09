import { Link } from 'react-router-dom';

export default function ResearcherLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-blue-600 text-white p-4">
        <div className="container mx-auto flex justify-between">
          <h1 className="font-bold text-xl">Researcher Dashboard</h1>
          <Link to="/" className="underline">Home</Link>
        </div>
      </header>
      <main className="container mx-auto p-4">{children}</main>
    </div>
  );
}