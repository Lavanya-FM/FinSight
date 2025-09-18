import axios from "axios";

const API_BASE_URL = process.env.REACT_APP_API_URL;

export const analyzeBankStatement = async (cibilScore, files) => {
  if (!API_BASE_URL) {
    throw new Error("REACT_APP_API_URL environment variable is not set");
  }

  const formData = new FormData();
  formData.append("cibil_score", cibilScore);
  Array.from(files).forEach(file => formData.append("files", file));

  try {
    const response = await axios.post(`${API_BASE_URL}/api/v1/analyze-document`, formData, {
      headers: { 
        "Content-Type": "multipart/form-data",
        "Authorization": `Bearer ${localStorage.getItem("access_token") || sessionStorage.getItem("access_token")}` // Assuming JWT token storage
      },
    });

    return response.data;
  } catch (error) {
    console.error("Error analyzing bank statement:", error);
    if (error.response) {
      throw new Error(`Analysis failed: ${error.response.data.detail || error.response.statusText}`);
    }
    throw new Error(`Network error: ${error.message}`);
  }
};

export default analyzeBankStatement;