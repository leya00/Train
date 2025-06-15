import { useEffect, useState } from "react";
import { AlertCircle, Clock10, TrainFront, Gauge } from "lucide-react";

export default function CommuterView() {
  const [countdown, setCountdown] = useState(300); // 5 minutes
  const [confidence, setConfidence] = useState(0.94);

  useEffect(() => {
    const interval = setInterval(() => {
      setCountdown((prev) => (prev > 0 ? prev - 1 : 0));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const formatTime = (seconds: number) => {
    const min = Math.floor(seconds / 60);
    const sec = seconds % 60;
    return `${min}m ${sec < 10 ? "0" : ""}${sec}s`;
  };

  const Card = ({ icon, title, children, className = "" }: any) => (
    <div className={`bg-white shadow-lg rounded-2xl p-6 ${className}`}>
      <div className="flex items-center space-x-4 mb-4">
        {icon}
        <h2 className="text-xl font-semibold">{title}</h2>
      </div>
      {children}
    </div>
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-6">
      <Card
        icon={<TrainFront className="text-blue-600 w-8 h-8" />}
        title="Next Train Countdown"
      >
        <p className="text-4xl font-bold text-center my-4">{formatTime(countdown)}</p>
        <p className="text-sm text-gray-600 text-center">Live update to the next arrival</p>
      </Card>

      <Card
        icon={<AlertCircle className="text-red-500 w-8 h-8" />}
        title="Personalized Alerts"
      >
        <p className="text-base">Minor delay detected on your usual train.</p>
        <p className="text-sm text-gray-500 mt-2">
          Confidence Level:{" "}
          <span className="font-semibold text-yellow-600">
            {Math.round(confidence * 100)}%
          </span>
        </p>
      </Card>

      <Card
        icon={<Clock10 className="text-green-600 w-8 h-8" />}
        title="Optimal Arrival Time"
        className="md:col-span-2"
      >
        <p className="text-base">
          Arrive at the station in <span className="font-semibold">3–5 minutes</span> to catch the next train comfortably.
        </p>
        <p className="text-sm text-gray-500 mt-2">
          Based on current train frequency and detection analytics.
        </p>
      </Card>

      <Card
        icon={<Gauge className="text-indigo-600 w-8 h-8" />}
        title="System Detection Confidence"
      >
        <div className="text-center text-3xl font-bold text-indigo-700 mt-2">
          {Math.round(confidence * 100)}%
        </div>
        <p className="text-sm text-gray-500 text-center mt-2">
          The model is operating with high reliability.
        </p>
      </Card>
    </div>
  );
}
