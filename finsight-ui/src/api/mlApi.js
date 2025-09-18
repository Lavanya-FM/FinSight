// finsight-ui/src/api/mlApi.js
const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";
// REACT_APP_API_URL=https://your-backend-service.onrender.com
export async function fetchPrediction(data) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return response.json();
}
