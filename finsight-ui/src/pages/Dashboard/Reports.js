// src/pages/Dashboard/Reports.js
import React, { useState, useEffect } from 'react';

const Reports = ({ backendUrl }) => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/reports`, {
        method: 'GET',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        const data = await response.json();
        setReports(data);
      }
    } catch (err) {
      console.error('Error fetching reports');
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div style={{ textAlign: 'center', padding: '40px' }}>Loading reports...</div>;

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
            <div key={report.id} style={{ background: '#fff', borderRadius: '10px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', padding: '20px' }}>
              <h3 style={{ color: '#2c3e50', marginBottom: '10px' }}>Report - {new Date(report.date).toLocaleDateString()}</h3>
              <p><strong style={{ color: '#3498db' }}>Summary:</strong> {report.summary}</p>
              <div>
                <strong style={{ color: '#3498db' }}>Insights:</strong>
                <ul style={{ margin: '10px 0', paddingLeft: '20px', color: '#666' }}>
                  {report.insights.map((insight, idx) => (
                    <li key={idx}>{insight}</li>
                  ))}
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