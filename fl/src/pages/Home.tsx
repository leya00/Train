import { Link } from 'react-router-dom';


export default function Home() {
  return (
    <div style={containerStyle}>
      <header style={headerStyle}>
        <h1 style={titleStyle}>TrainTrack Optimizer</h1>
        <p style={subtitleStyle}>
          Sophisticated transit analytics platform providing accurate, real-time train information for researchers and commuters
        </p>
      </header>

      <div style={cardsContainerStyle}>
        {/* Researcher Card */}
        <div style={cardStyle}>
          <h2 style={cardTitleStyle}>Researcher Mode</h2>
          <p style={cardTextStyle}>
            Upload and analyze train detection videos, visualize frame-by-frame detections, and compare actual vs expected schedules with comprehensive statistics.
          </p>
          <Link
            to="/researcher"
            style={blueButtonStyle}
            onMouseEnter={e => (e.currentTarget.style.backgroundColor = hoverBlueButtonStyle.backgroundColor)}
            onMouseLeave={e => (e.currentTarget.style.backgroundColor = blueButtonStyle.backgroundColor)}
          >
            Enter Researcher Mode
          </Link>
        </div>

        {/* Commuter Card */}
        <div style={cardStyle}>
          <h2 style={cardTitleStyle}>Commuter Mode</h2>
          <p style={cardTextStyle}>
            Get real-time train arrival countdowns, personalized alerts for delays, and recommendations for optimal station arrival times.
          </p>
          <Link
            to="/commuter"
            style={greenButtonStyle}
            onMouseEnter={e => (e.currentTarget.style.backgroundColor = hoverGreenButtonStyle.backgroundColor)}
            onMouseLeave={e => (e.currentTarget.style.backgroundColor = greenButtonStyle.backgroundColor)}
          >
            Enter Commuter Mode
          </Link>
        </div>
      </div>
    </div>
  );
}

const containerStyle: React.CSSProperties = {
  minHeight: '100vh',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '2rem',
  backgroundColor: '#edf6ff',
  fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
};

const headerStyle: React.CSSProperties = {
  textAlign: 'center',
  maxWidth: '900px',
};

const titleStyle: React.CSSProperties = {
  fontSize: '3rem',
  fontWeight: '700',
  fontFamily: '"Jersey 25", sans-serif',
  letterSpacing: '0.15em',
  marginBottom: '0.5rem',
  color: '#111827',
};


const subtitleStyle: React.CSSProperties = {
  fontSize: '1.7rem',
  color: '#4b5563',
};

const cardsContainerStyle: React.CSSProperties = {
  display: 'flex',
  gap: '2rem',
  flexWrap: 'wrap',
  justifyContent: 'center',
  maxWidth: '2200px',
};

const cardStyle: React.CSSProperties = {
  backgroundColor: 'white',
  boxShadow: '0 10px 20px rgba(0, 0, 0, 0.1)',
  borderRadius: '16px',
  padding: '4rem', // increased from 2rem
  width: '500px',  // increased from maxWidth: 400px
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'space-between',
};


const cardTitleStyle: React.CSSProperties = {
  fontSize: '2.5rem', // increased from 1.5rem
  fontWeight: '600',
  fontFamily: '"Jersey 25", sans-serif',
  letterSpacing: '0.15em',
  marginBottom: '1.25rem',
  color: '#1f2937',
};


const cardTextStyle: React.CSSProperties = {
  fontSize: '1.3rem', // increased from 1rem
  color: '#4b5563',
  marginBottom: '2rem',
  lineHeight: 1.75,
};


const buttonStyleBase: React.CSSProperties = {
  padding: '0.75rem 1.5rem',
  fontSize: '1.5rem',
  fontWeight: '600',
  color: 'white',
  borderRadius: '8px',
  border: 'none',
  cursor: 'pointer',
  textDecoration: 'none',
  display: 'inline-block',
  textAlign: 'center',
  userSelect: 'none',
  transition: 'background-color 0.3s ease',
};

const blueButtonStyle: React.CSSProperties = {
  ...buttonStyleBase,
  backgroundColor: '#2563eb',
};

const greenButtonStyle: React.CSSProperties = {
  ...buttonStyleBase,
  backgroundColor: '#16a34a',
};

const hoverBlueButtonStyle: React.CSSProperties = {
  backgroundColor: '#1e40af',
};

const hoverGreenButtonStyle: React.CSSProperties = {
  backgroundColor: '#15803d',
};
