// src/components/ModelPerformance.tsx
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import '../styles/modelPerformance.css'; 


interface ModelMetrics {
  f1_score: number | null;
  detection_accuracy: number | null;
  trains_per_hour: number | null;
  false_positives: number | null;
  precision?: number | null;
  recall?: number | null;
  avg_confidence?: number | null;
}

interface ModelPerformanceProps {
  detectionResults?: any;
  refreshTrigger?: number;
}

const ModelPerformance: React.FC<ModelPerformanceProps> = ({ detectionResults, refreshTrigger }) => {
    const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
    const [loading, setLoading] = useState<boolean>(false);

    const fetchMetrics = async () => {
        try {
            setLoading(true);
            const response = await axios.get('http://localhost:8000/model_metrics');
            setMetrics(response.data);
        } catch (error) {
            console.error('Failed to fetch model metrics:', error);
            setMetrics(null);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        // If detection results are passed directly, use those
        if (detectionResults?.statistics) {
            const stats = detectionResults.statistics;
            setMetrics({
                f1_score: stats.f1_score || null,
                detection_accuracy: stats.detection_rate || null,
                trains_per_hour: stats.trains_per_hour || null,
                false_positives: stats.false_positives || null,
                precision: stats.precision || null,
                recall: stats.recall || null,
                avg_confidence: stats.avg_confidence || null
            });
        } else {
            // Otherwise fetch from API
            fetchMetrics();
        }
    }, [detectionResults, refreshTrigger]);

    return (
        <div className="model-performance-container">
            <h2 className="performance-title">Model Performance</h2>
            {loading ? (
                <div className="loading">Loading metrics...</div>
            ) : (
                <div className="metrics-grid">
                    <div className="metric-box">
                        <span className="metric-label">F1 Score</span>
                        <span className="metric-value">
                            {metrics?.f1_score !== null ? metrics?.f1_score.toFixed(2) : '-'}
                        </span>
                    </div>
                    <div className="metric-box">
                        <span className="metric-label">Detection Rate</span>
                        <span className="metric-value">
                            {metrics?.detection_accuracy !== null ? `${metrics?.detection_accuracy}%` : '-'}
                        </span>
                    </div>
                    
                    <div className="metric-box">
                        <span className="metric-label"> false positive</span>
                        <span className="metric-value">
                            {metrics?.false_positives !== null ? metrics?.false_positives : '-'}
                        </span>
                    </div>
                    <div className="metric-box">
                        <span className="metric-label">Precision</span>
                        <span className="metric-value">
                            {metrics?.precision !== undefined && metrics?.precision !== null ? metrics.precision.toFixed(2) : '-'} {/* Constant error*/}
                        </span>
                    </div>
                    <div className='metric-box'>
                        <span className="metric-label">Avg Confidence</span>
                        <span className="metric-value">
                            {metrics?.avg_confidence !== null ? metrics?.avg_confidence : '-'}
                        </span>
                    </div>
                    <div className="metric-box">
                        <span className="metric-label">Recall</span>
                        <span className="metric-value">
                            {metrics?.recall !== undefined && metrics?.recall !== null ? metrics.recall.toFixed(2) : '-'}  {/* Fixed error */}

                        </span>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ModelPerformance;
