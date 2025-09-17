// src/utils/api.js
import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

export const analyzeBankStatement = async (cibilScore, files) => {
  const formData = new FormData();
  formData.append("cibil_score", cibilScore);
  Array.from(files).forEach(file => formData.append("files", file));

  const response = await axios.post(`${API_BASE_URL}/api/analyze`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
};
export default analyzeBankStatement;