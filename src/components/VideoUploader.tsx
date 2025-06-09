import axios from 'axios';
import { ChangeEvent, useState, useRef } from 'react';
import '../styles/VideoUploader.css'
import { PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';
import ModelPerformance from './ModelPerformance';


type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

type DetectionResults = {
  statistics: {
    frames_with_trains: number;
    object_counts?: Record<string, number>;
    f1_score?: number;
    detection_accuracy?: number;
    trains_per_hour?: number;
    false_positives?: number;
    precision?: number;
    recall?: number;
    avg_confidence?: number;
    detection_rate?: number;
    detection_time_seconds?: number;
    detection_fps?: number;
  };
  video_path: string;
  schedule?: {
    train_id: number;
    expected_time: number;
    expected_timestamp: string;
    tolerance: number;
    detected: boolean;
    detection_times: number[];
  }[];
};

export default function VideoUploader() {
  const [video, setVideo] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [status, setStatus] = useState<UploadStatus>('idle');
  const [result, setResult] = useState('');
  const [detectionResults, setDetectionResults] = useState<DetectionResults | null>(null);
  const [schedule, setSchedule] = useState([]);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.8);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

  
  
  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) {
      if (preview) {
        URL.revokeObjectURL(preview);
      }

      const newFile = e.target.files[0];
      setVideo(newFile);
      setPreview(URL.createObjectURL(newFile));
      setResult('');
      setStatus('idle');
      setUploadProgress(0);
      setDetectionResults(null);
    }
  }

  function handleBrowseFiles() {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }

  function handleThresholdChange(e: ChangeEvent<HTMLSelectElement>) {
    setConfidenceThreshold(parseFloat(e.target.value));
  }

  async function handleDetect() {
    if (!video) return alert('Please select a video first.');

    const formData = new FormData();
    formData.append('video', video);
    formData.append('threshold', confidenceThreshold.toString());

    try {
      setStatus('uploading');
      setUploadProgress(0);

      console.log('Sending request to backend...');
      console.log('Video file:', video.name, 'Size:', video.size);
      console.log('Threshold:', confidenceThreshold);

      const response = await axios.post('http://localhost:8000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (e) => {
          const progress = e.total ? Math.round((e.loaded * 100) / e.total) : 0;
          setUploadProgress(progress);
        },
      });

      console.log('Response received:', response.data);
      console.log('Schedule data:', response.data.schedule);
      
      // Create the detection results object with the correct structure
      const detectionResults = {
        message: response.data.message,
        video_path: response.data.video_path,
        statistics: response.data.statistics,
        schedule: response.data.schedule || []
      };

      setStatus('success');
      setUploadProgress(100);
      setResult(response.data.message || 'Train detection complete.');
      setDetectionResults(detectionResults);
      setSchedule(detectionResults.schedule);
      setRefreshTrigger(prev => prev + 1);

    } catch (error: any) {
      console.error('Upload error details:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
        headers: error.response?.headers
      });
      
      let errorMessage = 'Error processing video.';
      if (error.response?.data?.details) {
        errorMessage = `Error: ${error.response.data.details}`;
      } else if (error.response?.data?.message) {
        errorMessage = `Error: ${error.response.data.message}`;
      }
      
      setStatus('error');
      setResult(errorMessage);
      setUploadProgress(0);
    }
  }

  // Add this console log to debug the schedule data in the render
  console.log('Current detectionResults:', detectionResults);
  console.log('Current schedule:', schedule);

  return (
    <div className="dashboard-container">

      
      <div className="dashboard-content">
        {/* Upload Section */}
        <div className="dashboard-card upload-section">
          <h2>Upload Video</h2>
          <div className="video-drop-area" onClick={handleBrowseFiles}>
            {preview ? (
              <video key={preview} controls className="video-preview">
                <source src={preview} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            ) : (
              <div className="drag-drop-placeholder">
                <div className="placeholder-text">Drag & Drop Video File</div>
              </div>
            )}
          </div>
          
          <input 
            type="file" 
            ref={fileInputRef}
            id="video-upload" 
            className="video-input" 
            accept="video/*" 
            onChange={handleFileChange} 
          />
          
          <div className="action-buttons">
            <button 
              className="browse-button" 
              onClick={handleBrowseFiles}
            >
              Browse Files
            </button>

           <button
              className="detect-button"
              onClick={handleDetect}
              disabled={!video || status === 'uploading'}
            >
              {status === 'uploading' ? 'Detecting...' : 'Detect Trains'}
            </button>
          </div>
          
          {status === 'uploading' && (
            <div className="progress-container">
              <div className="progress-bar-bg">
                <div
                  className="progress-bar-fill" style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
              <p className="progress-text">{uploadProgress}% uploaded</p>
            </div>
          )}
        </div>
        
        
        {/* Detection Info Section */}
        <div className="dashboard-card detection-info">
          <h2>Trains Detected</h2>
          <div className="detection-stats">
            <div className="stat-item">
              <span className="stat-label">Total Detected</span>
              <span className="stat-value">{detectionResults?.statistics?.frames_with_trains ?? 0}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Detection Speed</span>
              <span className="stat-value">{detectionResults?.statistics?.detection_fps !== undefined && detectionResults?.statistics?.detection_time_seconds !== undefined
                ? `${detectionResults.statistics.detection_fps.toFixed(2)} FPS (${detectionResults.statistics.detection_time_seconds.toFixed(2)}s)`
                : 'N/A'}
              </span>
            </div>

          </div>
          
    
          
          <div className="threshold-section">
            <h3>Change Threshold</h3>
            <div className="threshold-selector">
              <select 
                value={confidenceThreshold} 
                onChange={handleThresholdChange}
                className="threshold-dropdown"
              >
                <option value="0.1">0.1</option>
                <option value="0.2">0.2</option>
                <option value="0.3">0.3</option>
                <option value="0.4">0.4</option>
                <option value="0.5">0.5</option>
                <option value="0.6">0.6</option>
                <option value="0.7">0.7</option>
                <option value="0.8">0.8 (Default)</option>
                <option value="0.9">0.9</option>
                <option value="1.0">1.0</option>
              </select>
              <div className="dropdown-arrow">▼</div>
            </div>
          </div>
        </div>
        
        {/* Model Performance Section - Using the ModelPerformance component */}
        <ModelPerformance     
          detectionResults={detectionResults} 
          refreshTrigger={refreshTrigger} 
        />
      </div>
      
      {/* Results Tables */}
      <div className="results-section">
        <div className="schedule-results">
          <h2>Train Schedule & Detection Results</h2>
          <table className="results-table">
            <thead>
              <tr>
                <th>Train ID</th>
                <th>Scheduled Time</th>
                <th>Detection Status</th>
                <th>Actual Detection Time</th>
                <th>Time Difference</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {detectionResults?.schedule ? (
                detectionResults.schedule.map((entry, index) => {
                  // Format the actual detection time as HH:MM:SS
                  const actualDetectionTime = entry.detection_times && entry.detection_times.length > 0 
                    ? new Date(entry.detection_times[0] * 1000).toISOString().substr(11, 8)
                    : '-';
                  
                  // Calculate time difference in seconds with proper rounding
                  const timeDiff = entry.detection_times && entry.detection_times.length > 0
                    ? Math.round(Math.abs(entry.detection_times[0] - entry.expected_time))
                    : '-';
                  
                  // Get confidence from the first detection
                  const confidence = entry.detection_times && entry.detection_times.length > 0
                    ? 'High'  // You might want to get actual confidence from backend
                    : '-';
                  
                  return (
                    <tr key={`schedule-row-${index}`} className={entry.detected ? 'detected' : 'missed'}>
                      <td>Train {entry.train_id}</td>
                      <td>{entry.expected_timestamp}</td>
                      <td>
                        <span className={`status-badge ${entry.detected ? 'detected' : 'missed'}`}>
                          {entry.detected ? 'Detected' : 'Not Detected'}
                        </span>
                      </td>
                      <td>{actualDetectionTime}</td>
                      <td>{timeDiff !== '-' ? `${timeDiff}s` : '-'}</td>
                      <td>
                        {entry.detected ? (
                          <span className={`confidence-badge ${confidence === 'High' ? 'high' : 'low'}`}>
                            {confidence}
                          </span>
                        ) : '-'}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={6} className="no-data">No schedule data available</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        
        <div className="objects-detected">
          <h2>Objects detected in a day</h2>
          <div className="chart-container">
            {detectionResults?.statistics?.object_counts ? (
              <>
                <PieChart width={300} height={300}>
                  <Pie
                    data={Object.entries(detectionResults.statistics.object_counts).map(([name, value]) => ({
                      name,
                      value
                    }))}
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                    nameKey="name"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {Object.entries(detectionResults.statistics.object_counts).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
                <div className="object-stats">
                  <h3>Detection Breakdown</h3>
                  <ul>
                    {Object.entries(detectionResults.statistics.object_counts).map(([name, count], index) => {
                      const total = Object.values(detectionResults.statistics.object_counts ?? {}).reduce((a, b) => a + b, 0);
                      const percentage = ((count / total) * 100).toFixed(1);
                      return (
                        <li key={name} style={{ color: COLORS[index % COLORS.length] }}>
                          {name}: {count} ({percentage}%)
                        </li>
                      );
                    })}
                  </ul>
                </div>
              </>
            ) : (
              <div className="empty-chart-placeholder">
                <p>No detection data available</p>
              </div>
            )}
          </div>
        </div>
      </div>
      
      <div className="dashboard-footer">
        <div className="project-info">2025 RuralAI capstone Project</div>
      </div>
    </div>
  );
}
