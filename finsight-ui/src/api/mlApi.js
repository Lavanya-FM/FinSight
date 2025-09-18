const API_BASE_URL = process.env.REACT_APP_API_URL;

export async function fetchPrediction(data) {
  if (!API_BASE_URL) {
    throw new Error("REACT_APP_API_URL environment variable is not set");
  }
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "Authorization": `Bearer ${localStorage.getItem("access_token")}` // Assuming JWT is stored in localStorage
      },
      body: JSON.stringify(data),
    });
    if (!response.ok) {
      throw new Error(`Prediction request failed: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error fetching prediction:", error);
    throw error;
  }
}