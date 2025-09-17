import axios from "axios";

const API_BASE = "http://localhost:8000"; // Update with backend URL

// Call ML API with CIBIL + uploaded document
export const analyzeBankStatement = async (cibilScore, file) => {
  const formData = new FormData();
  formData.append("cibil_score", cibilScore);
  formData.append("file", file);

  const res = await axios.post(`${API_BASE}/analyze/`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};
