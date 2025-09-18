import React, { useState, useEffect } from 'react';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.REACT_APP_SUPABASE_URL,
  process.env.REACT_APP_SUPABASE_ANON_KEY
);

const Reports = ({ backendUrl }) => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    console.log('Backend URL:', backendUrl);
    fetchReports();
  }, []);

  const fetchReports = async () => {
    try {
      const { data: { session }, error: sessionError } = await supabase.auth.getSession();
      if (sessionError || !session) {
        throw new Error('No user session found. Please log in.');
      }
      const token = session.access_token;
      console.log('Fetching reports from:', `${backendUrl}/api/v1/reports`);
      console.log('Using token:', token.substring(0, 20) + '...');

      const response = await fetch(`${backendUrl}/api/v1/reports`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', Array.from(response.headers.entries()));

      if (!response.ok) {
        let errorDetail;
        try {
          errorDetail = await response.json();
          console.error('Error response body:', errorDetail);
        } catch {
          errorDetail = { detail: await response.text() };
          console.error('Non-JSON response body:', errorDetail.detail);
        }
        throw new Error(`HTTP error! status: ${response.status}, detail: ${errorDetail.detail || response.statusText}`);
      }

      const data = await response.json();
      console.log('Reports fetched:', data);
      setReports(data);
    } catch (err) {
      console.error('Error fetching reports:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div style={{ textAlign: 'center', padding: '40px' }}>Loading reports...</div>;
  if (error) return <div style={{ textAlign: 'center', padding: '40px', color: 'red' }}>Error: {error}</div>;

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto', padding: '40px 20px' }}>
      <h1 style={{ color: '#2c3e50', marginBottom: '30px', fontSize: '2rem', fontWeight: 700 }}>Reports</h1>
      {reports.length === 0 ? (
        <div style={{ background: '#fff', borderRadius: '10px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', padding: '30px', textAlign: 'center', color: '#666' }}>
          No reports generated yet. Upload a document to get started.
        </div>
      ) : (
        <div style={{ display: 'grid', gap: '20px' }}>
          {reports.map((report) => (
            <div key={report.report_id} style={{ background: '#fff', borderRadius: '10px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', padding: '20px' }}>
              <h3 style={{ color: '#2c3e50', marginBottom: '10px' }}>
                Report - {new Date(report.generated_at).toLocaleDateString()}
              </h3>
              <p><strong style={{ color: '#3498db' }}>File:</strong> {report.report_details?.file_analysis?.file_name || 'Unknown'}</p>
              <p><strong style={{ color: '#3498db' }}>Applicant:</strong> {report.report_details?.file_analysis?.applicant_name || 'Unknown'}</p>
              <p><strong style={{ color: '#3498db' }}>Decision:</strong> {report.action}</p>
              <p><strong style={{ color: '#3498db' }}>Reason:</strong> {report.reason}</p>
              <p><strong style={{ color: '#3498db' }}>Confidence:</strong> {(report.confidence * 100).toFixed(1)}%</p>
              <div>
                <strong style={{ color: '#3498db' }}>Financial Metrics:</strong>
                <ul style={{ margin: '10px 0', paddingLeft: '20px', color: '#666' }}>
                  <li>Monthly Income: ₹{report.report_details?.financial_metrics?.monthly_income?.toLocaleString() || 'N/A'}</li>
                  <li>CIBIL Score: {report.report_details?.financial_metrics?.cibil_score || 'N/A'}</li>
                  <li>Savings Rate: {report.report_details?.financial_metrics?.savings_rate || 'N/A'}%</li>
                  <li>Debt-to-Income Ratio: {report.report_details?.financial_metrics?.debt_to_income_ratio || 'N/A'}%</li>
                </ul>
              </div>
              <div>
                <strong style={{ color: '#3498db' }}>Risk Assessment:</strong>
                <ul style={{ margin: '10px 0', paddingLeft: '20px', color: '#666' }}>
                  <li>Risk Category: {report.report_details?.risk_assessment?.risk_category || 'N/A'}</li>
                  <li>Overall Risk Score: {report.report_details?.risk_assessment?.overall_risk_score || 'N/A'}</li>
                </ul>
              </div>
              <div>
                <strong style={{ color: '#3498db' }}>Recommendations:</strong>
                <ul style={{ margin: '10px 0', paddingLeft: '20px', color: '#666' }}>
                  {report.report_details?.recommendations?.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  )) || <li>No recommendations available</li>}
                </ul>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Reports;
