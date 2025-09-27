import { useState } from 'react';
import VideoUploader from '../components/VideoUploader';

export default function Researcher() {
  const [uploadResult, setUploadResult] = useState(null);

  const handleUploadSuccess = (data: any) => {
    setUploadResult(data);
    // Handle result: show detection, schedule, etc.
  };

  return (
    <div>
      <h1>Researcher View</h1>
      <VideoUploader onUploadSuccess={handleUploadSuccess} />
      {uploadResult && <pre>{JSON.stringify(uploadResult, null, 2)}</pre>}
    </div>
  );
}
