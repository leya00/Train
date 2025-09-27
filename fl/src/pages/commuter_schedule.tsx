import CommuterLayout from '../layouts/CommuterLayout';

export default function Commuter() {
  return (
    <CommuterLayout>
      <div className="p-6 text-center">
        <h2 className="text-2xl font-semibold mb-4">Next Train Status</h2>
        <p>Placeholder for arrival/delay info.</p>
      </div>
    </CommuterLayout>
  );
}