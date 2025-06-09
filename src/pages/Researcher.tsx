import { useState, useRef } from 'react';
import axios from 'axios';
import ResearcherLayout from '../layouts/ResearcherLayout';
import ModelPerformance from '../components/ModelPerformance';
import { UploadStatus, DetectionResults } from '../lib/types';
import '../styles/upload.css'; // Ensure this path is correct

export default function Researcher() {
  const [video, setVideo] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [status, setStatus] = useState<UploadStatus>('idle');
  const [result, setResult] = useState('');
  const [detectionResults, setDetectionResults] = useState<DetectionResults | null>(null);
  const [schedule, setSchedule] = useState([]);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.8);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setVideo(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setDetectionResults(null);
      setResult('');
    }
  };

  const handleBrowseFiles = () => {
    fileInputRef.current?.click();
  };

  const handleDetect = async () => {
    if (!video) return;

    const formData = new FormData();
    formData.append('video', video);
    formData.append('threshold', confidenceThreshold.toString());

    try {
      setStatus('uploading');
      setUploadProgress(0);
      const response = await axios.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const percent = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 1));
          setUploadProgress(percent);
        },
      });

      const { statistics, schedule, result } = response.data;
      setDetectionResults(statistics);
      setSchedule(schedule);
      setResult(result);
      setStatus('success');
    } catch (error) {
      console.error('Error uploading file', error);
      setStatus('error');
    }
  };

  return (
    <ResearcherLayout>
      <div className="upload-container">
        <div className="upload-card">
          <h1 className="upload-title">Upload Training Video</h1>
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="hidden"
            ref={fileInputRef}
          />
          <button onClick={handleBrowseFiles} className="upload-button">
            Browse Files
          </button>
          {preview && <video src={preview} controls className="upload-preview" />}
          <div className="upload-controls">
            <label>Confidence Threshold:</label>
            <select
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              className="upload-select"
            >
              {[0.5, 0.6, 0.7, 0.8, 0.9].map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
            <button
              onClick={handleDetect}
              className="detect-button"
              disabled={!video || status === 'uploading'}
            >
              {status === 'uploading' ? `Uploading... ${uploadProgress}%` : 'Detect Trains'}
            </button>
          </div>
        </div>

        {detectionResults && (
          <>
            <div className="upload-card">
              <h3 className="upload-title">Detection Summary</h3>
              <pre className="upload-summary">{JSON.stringify(detectionResults, null, 2)}</pre>
            </div>

            <ModelPerformance detectionResults={detectionResults} />

            <div className="upload-card">
              <h3 className="upload-title">Detected Schedule</h3>
              <table className="upload-table">
                <thead>
                  <tr>
                    <th>Train ID</th>
                    <th>Detected Time</th>
                    <th>Expected Time</th>
                    <th>Time Difference</th>
                  </tr>
                </thead>
                <tbody>
                  {schedule.map((item: any, index: number) => (
                    <tr key={index}>
                      <td>{item.trainId}</td>
                      <td>{item.detectedTime}</td>
                      <td>{item.expectedTime}</td>
                      <td>{item.timeDifference} sec</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </ResearcherLayout>
  );
}
