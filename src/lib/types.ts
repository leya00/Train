export type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

export type DetectionResults = {
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
