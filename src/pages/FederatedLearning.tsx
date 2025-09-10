import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';

interface TrainingStatus {
  is_running: boolean;
  current_round: number;
  total_rounds: number;
  clients_connected: number;
  start_time: string | null;
  logs: string[];
}

interface FederatedResult {
  type: string;
  name: string;
  path: string;
  size: string;
}

export default function FederatedLearning() {
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    is_running: false,
    current_round: 0,
    total_rounds: 0,
    clients_connected: 0,
    start_time: null,
    logs: []
  });
  
  const [federatedResults, setFederatedResults] = useState<FederatedResult[]>([]);
  const [isStartingTraining, setIsStartingTraining] = useState(false);
  const [isLoadingResults, setIsLoadingResults] = useState(false);
  
  const [trainingConfig, setTrainingConfig] = useState({
    rounds: 4,
    clients: 4,  // Changed to 4 clients to match your teammate's setup
    dataset: 'All 4 Datasets',  // Updated to reflect all datasets
    epochs: 3  // Changed to 3 epochs to match your teammate's setup
  });

  // Poll training status
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:5000/training-status');
        const status = await response.json();
        setTrainingStatus(status);
      } catch (error) {
        console.error('Failed to fetch training status:', error);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const handleStartTraining = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsStartingTraining(true);

    const formData = new FormData();
    formData.append('rounds', trainingConfig.rounds.toString());
    formData.append('clients', trainingConfig.clients.toString());
    formData.append('dataset', trainingConfig.dataset);
    formData.append('epochs', trainingConfig.epochs.toString());

    try {
      const response = await fetch('http://localhost:5000/start-federated-training', {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      if (result.success) {
        alert('Federated learning started successfully!');
      } else {
        alert(`Failed to start training: ${result.error}`);
      }
    } catch (error) {
      alert(`Training start failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsStartingTraining(false);
    }
  };

  const loadFederatedResults = async () => {
    setIsLoadingResults(true);
    try {
      const response = await fetch('http://localhost:5000/federated-results');
      const data = await response.json();
      setFederatedResults(data.results || []);
    } catch (error) {
      console.error('Failed to load federated results:', error);
      setFederatedResults([]);
    } finally {
      setIsLoadingResults(false);
    }
  };

  const getProgressPercentage = () => {
    if (trainingStatus.total_rounds === 0) return 0;
    return (trainingStatus.current_round / trainingStatus.total_rounds) * 100;
  };

  return (
    <div style={containerStyle}>
      <header style={headerStyle}>
        <h1 style={titleStyle}> Federated Learning Dashboard</h1>
        <p style={subtitleStyle}>
          Collaborative train detection model training with privacy-preserving federated learning
        </p>
        <Link to="/" style={backButtonStyle}>
          ← Back to Home
        </Link>
      </header>

      <div style={contentStyle}>
        {/* Training Status Section */}
        <div style={sectionStyle}>
          <h2 style={sectionTitleStyle}>Training Status</h2>
          <div style={statusCardStyle}>
            <div style={statusHeaderStyle}>
              <div style={statusIndicatorStyle(trainingStatus.is_running)}></div>
              <strong style={statusTextStyle}>
                {trainingStatus.is_running ? 'Running' : 'Stopped'}
              </strong>
            </div>
            
            <div style={progressBarStyle}>
              <div 
                style={{
                  ...progressFillStyle,
                  width: `${getProgressPercentage()}%`
                }}
              ></div>
            </div>
            
            <div style={statusDetailsStyle}>
              <p>Round {trainingStatus.current_round} of {trainingStatus.total_rounds}</p>
              <p>Clients Connected: {trainingStatus.clients_connected}</p>
              <p>Start Time: {trainingStatus.start_time || '-'}</p>
            </div>
          </div>
        </div>

        {/* Federated Learning Configuration */}
        <div style={sectionStyle}>
          <h2 style={sectionTitleStyle}> Federated Learning Setup</h2>
          <form onSubmit={handleStartTraining} style={formStyle}>
            <div style={gridStyle}>
              <div style={formGroupStyle}>
                <label htmlFor="rounds" style={labelStyle}>Training Rounds:</label>
                <input
                  type="number"
                  id="rounds"
                  value={trainingConfig.rounds}
                  onChange={(e) => setTrainingConfig(prev => ({
                    ...prev,
                    rounds: parseInt(e.target.value) || 4
                  }))}
                  min="1"
                  max="100"
                  style={inputStyle}
                />
              </div>
              
              <div style={formGroupStyle}>
                <label htmlFor="clients" style={labelStyle}>Number of Clients:</label>
                <input
                  type="number"
                  id="clients"
                  value={trainingConfig.clients}
                  onChange={(e) => setTrainingConfig(prev => ({
                    ...prev,
                    clients: parseInt(e.target.value) || 2
                  }))}
                  min="1"
                  max="10"
                  style={inputStyle}
                />
              </div>
              
              <div style={formGroupStyle}>
                <label htmlFor="epochs" style={labelStyle}>Training Epochs:</label>
                <input
                  type="number"
                  id="epochs"
                  value={trainingConfig.epochs}
                  onChange={(e) => setTrainingConfig(prev => ({
                    ...prev,
                    epochs: parseInt(e.target.value) || 5
                  }))}
                  min="1"
                  max="50"
                  style={inputStyle}
                />
              </div>
              
              <div style={formGroupStyle}>
                <label htmlFor="dataset" style={labelStyle}>Dataset Source:</label>
                <select
                  id="dataset"
                  value={trainingConfig.dataset}
                  onChange={(e) => setTrainingConfig(prev => ({
                    ...prev,
                    dataset: e.target.value
                  }))}
                  style={inputStyle}
                >
                  <option value="data/labelfront1">Front Camera Data 1</option>
                  <option value="data/labelfront2">Front Camera Data 2</option>
                  <option value="data/labelback1">Back Camera Data 1</option>
                  <option value="data/labelback2">Back Camera Data 2</option>
                </select>
              </div>
            </div>
            
            <button 
              type="submit" 
              disabled={isStartingTraining || trainingStatus.is_running}
              style={{
                ...buttonStyle,
                ...buttonSuccessStyle,
                ...(isStartingTraining || trainingStatus.is_running ? disabledButtonStyle : {})
              }}
            >
              {isStartingTraining ? 'Starting...' : 'Start Federated Learning'}
            </button>
          </form>
        </div>

        {/* Federated Learning Results */}
        <div style={sectionStyle}>
          <h2 style={sectionTitleStyle}> Training Results</h2>
          <div style={resultsHeaderStyle}>
            <p style={resultsDescriptionStyle}>
              View the results of your federated learning training session, including the global aggregated model and individual client models.
            </p>
            <button 
              onClick={loadFederatedResults}
              disabled={isLoadingResults}
              style={{
                ...buttonStyle,
                ...(isLoadingResults ? disabledButtonStyle : {})
              }}
            >
              {isLoadingResults ? 'Loading...' : 'Refresh Results'}
            </button>
          </div>
          
          {federatedResults.length > 0 ? (
            <div style={resultsGridStyle}>
              {federatedResults.map((result, index) => (
                <div key={index} style={{
                  ...resultCardStyle,
                  ...(result.type === 'final_model' ? finalModelCardStyle : clientModelCardStyle)
                }}>
                  <h3 style={resultTitleStyle}>{result.name}</h3>
                  <div style={resultDetailsStyle}>
                    <p><strong>Type:</strong> {result.type === 'final_model' ? 'Global Aggregated' : 'Client Model'}</p>
                    <p><strong>Path:</strong> {result.path}</p>
                    <p><strong>Size:</strong> {result.size}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={noResultsStyle}>
              <p>No training results found yet. Start federated learning to see results here.</p>
            </div>
          )}
        </div>

        {/* Quick Actions */}
        <div style={sectionStyle}>
          <h2 style={sectionTitleStyle}>⚡ Quick Actions</h2>
          <div style={gridStyle}>
            <div style={cardStyle}>
              <h3 style={cardTitleStyle}>View Raw Results</h3>
              <p style={cardTextStyle}>
                Check the raw federated learning output files and logs
              </p>
              <a 
                href="http://localhost:5000/federated-results" 
                target="_blank" 
                rel="noopener noreferrer"
                style={buttonStyle}
              >
                View Raw Results
              </a>
            </div>
            
            <div style={cardStyle}>
              <h3 style={cardTitleStyle}>Training Logs</h3>
              <p style={cardTextStyle}>
                Monitor training progress and debug issues
              </p>
              <button 
                onClick={async () => {
                  try {
                    const response = await fetch('http://localhost:5000/logs');
                    const logs = await response.json();
                    console.log('Training logs:', logs);
                    alert('Logs loaded! Check console for details.');
                  } catch (error) {
                    alert('Failed to load logs');
                  }
                }}
                style={buttonStyle}
              >
                View Logs
              </button>
            </div>
            
            <div style={cardStyle}>
              <h3 style={cardTitleStyle}>Debug Output</h3>
              <p style={cardTextStyle}>
                Check what's actually in your output directory
              </p>
              <button 
                onClick={async () => {
                  try {
                    const response = await fetch('http://localhost:5000/debug-output');
                    const debug = await response.json();
                    console.log('Debug output:', debug);
                    alert('Debug info loaded! Check console for details.');
                  } catch (error) {
                    alert('Failed to load debug info');
                  }
                }}
                style={buttonStyle}
              >
                Debug Output
              </button>
            </div>
            
            <div style={cardStyle}>
              <h3 style={cardTitleStyle}>Test Training</h3>
              <p style={cardTextStyle}>
                Manually test if training works
              </p>
              <button 
                onClick={async () => {
                  try {
                    const response = await fetch('http://localhost:5000/test-training');
                    const test = await response.json();
                    console.log('Test training:', test);
                    alert('Training test completed! Check console for details.');
                  } catch (error) {
                    alert('Failed to run training test');
                  }
                }}
                style={buttonStyle}
              >
                Test Training
              </button>
            </div>
            
            <div style={cardStyle}>
              <h3 style={cardTitleStyle}>Output Directory</h3>
              <p style={cardTextStyle}>
                Access the output folder with all training results
              </p>
              <a 
                href="http://localhost:5000/static/output" 
                target="_blank" 
                rel="noopener noreferrer"
                style={buttonStyle}
              >
                Open Output Folder
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Styles
const containerStyle: React.CSSProperties = {
  minHeight: '100vh',
  backgroundColor: '#edf6ff',
  fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
};

const headerStyle: React.CSSProperties = {
  textAlign: 'center',
  padding: '3rem 2rem',
  backgroundColor: 'white',
  boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)',
};

const titleStyle: React.CSSProperties = {
  fontSize: '3rem',
  fontWeight: '700',
  fontFamily: '"Jersey 25", sans-serif',
  letterSpacing: '0.15em',
  marginBottom: '1rem',
  color: '#111827',
};

const subtitleStyle: React.CSSProperties = {
  fontSize: '1.5rem',
  color: '#4b5563',
  marginBottom: '2rem',
};

const backButtonStyle: React.CSSProperties = {
  display: 'inline-block',
  padding: '0.75rem 1.5rem',
  backgroundColor: '#6b7280',
  color: 'white',
  textDecoration: 'none',
  borderRadius: '8px',
  fontWeight: '600',
  transition: 'background-color 0.3s ease',
};

const contentStyle: React.CSSProperties = {
  maxWidth: '1200px',
  margin: '0 auto',
  padding: '2rem',
};

const sectionStyle: React.CSSProperties = {
  marginBottom: '3rem',
  padding: '2rem',
  border: '1px solid #e1e8ed',
  borderRadius: '16px',
  backgroundColor: 'white',
  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.05)',
};

const sectionTitleStyle: React.CSSProperties = {
  color: '#2c3e50',
  marginBottom: '1.5rem',
  fontSize: '1.8rem',
  borderBottom: '2px solid #3498db',
  paddingBottom: '0.5rem',
};

const statusCardStyle: React.CSSProperties = {
  backgroundColor: '#f8f9fa',
  padding: '1.5rem',
  borderRadius: '12px',
  borderLeft: '5px solid #3498db',
};

const statusHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  marginBottom: '1rem',
};

const statusIndicatorStyle = (isRunning: boolean): React.CSSProperties => ({
  width: '12px',
  height: '12px',
  borderRadius: '50%',
  marginRight: '10px',
  backgroundColor: isRunning ? '#27ae60' : '#e74c3c',
  animation: isRunning ? 'pulse 2s infinite' : 'none',
});

const statusTextStyle: React.CSSProperties = {
  fontSize: '1.2rem',
  fontWeight: '600',
};

const progressBarStyle: React.CSSProperties = {
  width: '100%',
  height: '20px',
  backgroundColor: '#ecf0f1',
  borderRadius: '10px',
  overflow: 'hidden',
  margin: '1rem 0',
};

const progressFillStyle: React.CSSProperties = {
  height: '100%',
  background: 'linear-gradient(90deg, #3498db, #2ecc71)',
  transition: 'width 0.3s ease',
};

const statusDetailsStyle: React.CSSProperties = {
  color: '#4b5563',
  fontSize: '1rem',
};

const formStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '1rem',
};

const formGroupStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '0.5rem',
};

const labelStyle: React.CSSProperties = {
  fontWeight: '600',
  color: '#34495e',
  fontSize: '1rem',
};

const inputStyle: React.CSSProperties = {
  padding: '0.75rem',
  border: '2px solid #ddd',
  borderRadius: '8px',
  fontSize: '1rem',
  transition: 'border-color 0.3s',
};

const gridStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
  gap: '1.5rem',
  marginTop: '1.5rem',
};

const buttonStyle: React.CSSProperties = {
  padding: '0.75rem 1.5rem',
  backgroundColor: '#3498db',
  color: 'white',
  border: 'none',
  borderRadius: '8px',
  fontSize: '1rem',
  fontWeight: '600',
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  textDecoration: 'none',
  display: 'inline-block',
  textAlign: 'center',
};

const buttonSuccessStyle: React.CSSProperties = {
  backgroundColor: '#27ae60',
};

const disabledButtonStyle: React.CSSProperties = {
  backgroundColor: '#bdc3c7',
  cursor: 'not-allowed',
};

const alertStyle: React.CSSProperties = {
  padding: '1rem',
  borderRadius: '8px',
  marginTop: '1rem',
};

const alertSuccessStyle: React.CSSProperties = {
  backgroundColor: '#d4edda',
  color: '#155724',
  border: '1px solid #c3e6cb',
};

const alertErrorStyle: React.CSSProperties = {
  backgroundColor: '#f8d7da',
  color: '#721c24',
  border: '1px solid #f5c6cb',
};

const cardStyle: React.CSSProperties = {
  backgroundColor: 'white',
  padding: '1.5rem',
  borderRadius: '12px',
  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  borderTop: '4px solid #3498db',
  display: 'flex',
  flexDirection: 'column',
  gap: '1rem',
};

const cardTitleStyle: React.CSSProperties = {
  fontSize: '1.3rem',
  fontWeight: '600',
  color: '#2c3e50',
  margin: 0,
};

const cardTextStyle: React.CSSProperties = {
  color: '#4b5563',
  lineHeight: 1.6,
  margin: 0,
  flex: 1,
};

const resultsHeaderStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: '1.5rem',
};

const resultsDescriptionStyle: React.CSSProperties = {
  fontSize: '1.1rem',
  color: '#555',
  margin: 0,
  flex: 1,
};

const resultsGridStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
  gap: '1.5rem',
};

const resultCardStyle: React.CSSProperties = {
  backgroundColor: '#f8f9fa',
  padding: '1.5rem',
  borderRadius: '12px',
  borderLeft: '5px solid #3498db',
  boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)',
};

const finalModelCardStyle: React.CSSProperties = {
  borderLeft: '5px solid #27ae60', // Green for final model
};

const clientModelCardStyle: React.CSSProperties = {
  borderLeft: '5px solid #e74c3c', // Red for client model
};

const resultTitleStyle: React.CSSProperties = {
  fontSize: '1.2rem',
  fontWeight: '600',
  color: '#2c3e50',
  marginBottom: '0.5rem',
};

const resultDetailsStyle: React.CSSProperties = {
  color: '#4b5563',
  fontSize: '0.9rem',
};

const noResultsStyle: React.CSSProperties = {
  textAlign: 'center',
  padding: '2rem',
  color: '#888',
};

// Add CSS animation for pulse effect
const style = document.createElement('style');
style.textContent = `
  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
  }
`;
document.head.appendChild(style);
