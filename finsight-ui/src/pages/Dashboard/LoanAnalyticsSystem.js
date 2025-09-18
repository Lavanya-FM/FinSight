import React, { useState, useEffect, useRef } from 'react';
import { Upload, FileText, CreditCard, CheckCircle, Download, TrendingUp, FileDown, Eye, Trash2, Calendar, DollarSign, Shield, Target } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area, RadarChart, PolarGrid, PolarAngleAxis, Radar } from 'recharts';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import { autoTable } from 'jspdf-autotable'; // Import autoTable correctly
import { createClient } from '@supabase/supabase-js';

// Initialize Supabase client
const supabase = createClient(
  process.env.REACT_APP_SUPABASE_URL,
  process.env.REACT_APP_SUPABASE_ANON_KEY
);

// Chart color schemes
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];
const RISK_COLORS = { 'Low Risk': '#22C55E', 'Medium Risk': '#F59E0B', 'High Risk': '#EF4444' };

// Enhanced API configuration for FastAPI backend
const API_BASE_URL = 'http://localhost:8000';
const API_ENDPOINTS = {
  ANALYZE_DOCUMENT: `${API_BASE_URL}/api/v1/analyze-document`,
  HEALTH_CHECK: `${API_BASE_URL}/api/v1/health`,
  GET_ANALYSIS: `${API_BASE_URL}/api/v1/analysis`,
  GET_REPORTS: `${API_BASE_URL}/api/v1/reports`,
  SAVE_REPORT: `${API_BASE_URL}/api/v1/save-report`,
};

// Add debug log to verify endpoint URLs
console.log('API_ENDPOINTS:', API_ENDPOINTS);

// Health check function
const checkAPIHealth = async () => {
  try {
    console.log('Checking API health at:', API_ENDPOINTS.HEALTH_CHECK);
    const response = await fetch(API_ENDPOINTS.HEALTH_CHECK, {
      method: 'GET',
    });
    const responseBody = await response.json().catch(() => ({}));
    console.log('Health check response:', response.status, response.statusText, responseBody);
    return response.ok;
  } catch (error) {
    console.warn('API health check failed:', error);
    return false;
  }
};

// Real-time analysis function with enhanced error handling
const analyzeFile = async (file, cibilScore) => {
  const maxFileSize = 50 * 1024 * 1024;
  const allowedTypes = [
    'application/pdf',
    'text/csv',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'text/plain'
  ];

  if (file.size > maxFileSize) {
    throw new Error(`File ${file.name} is too large. Maximum size is 50MB.`);
  }

  if (!allowedTypes.includes(file.type)) {
    throw new Error(`File type ${file.type} is not supported.`);
  }
  
  const { data: { session }, error: sessionError } = await supabase.auth.getSession();
  if (sessionError || !session) {
    throw new Error('No user session found. Please log in.');
  }
  const token = session.access_token;
  console.log('Analyzing file with token:', token.substring(0, 20) + '...');

  const formData = new FormData();
  formData.append('files', file);
  formData.append('cibil_score', cibilScore.toString());
  formData.append('analysis_type', 'comprehensive');
  formData.append('include_charts', 'true');

  try {
    console.log('Sending request to:', API_ENDPOINTS.ANALYZE_DOCUMENT);
    console.log('FormData contents:', { file: file.name, cibil_score: cibilScore });

    const response = await fetch(API_ENDPOINTS.ANALYZE_DOCUMENT, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
      body: formData,
    });

    console.log('Response status:', response.status);
    console.log('Response headers:', [...response.headers.entries()]);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error('Error response body:', errorData);
      throw new Error(
        errorData.detail ||
        errorData.message ||
        `Analysis failed with status ${response.status}: ${response.statusText}`
      );
    }

    const result = await response.json();
    console.log('Analysis result:', result);

    if (!result.financial_metrics || !result.decision_summary || !result.risk_assessment) {
      throw new Error('Invalid response structure from analysis service');
    }
    
    // Calculate new insights
    const incomeStabilityIndex = calculateIncomeStabilityIndex(result);
    const expenseVolatilityScore = calculateExpenseVolatilityScore(result);
    const loanAffordabilityRatio = calculateLoanAffordabilityRatio(result);
    
    return {
      ...result,
      id: Date.now(), // Temporary ID for frontend
      generated_at: new Date().toISOString(),
      client_processed: true,
      additional_insights: {
        income_stability_index: incomeStabilityIndex,
        expense_volatility_score: expenseVolatilityScore,
        loan_affordability_ratio: loanAffordabilityRatio,
        insights_summary: generateInsightsSummary({
          incomeStabilityIndex,
          expenseVolatilityScore,
          loanAffordabilityRatio,
          cibilScore: result.financial_metrics.cibil_score,
          riskCategory: result.risk_assessment.risk_category,
        }),
      },
    };
  } catch (error) {
    console.error('Analysis API error:', error);
    throw error;
  }
};

// Calculate Income Stability Index
const calculateIncomeStabilityIndex = (result) => {
  const variability = result.financial_metrics?.income_variability || 0;
  const transactionConsistency = result.detailed_analysis?.transaction_frequency || 'Moderate';
  let stabilityScore = 100 - variability;
  if (transactionConsistency === 'High') stabilityScore *= 1.1;
  else if (transactionConsistency === 'Low') stabilityScore *= 0.9;
  return Math.min(Math.max(Math.round(stabilityScore), 0), 100);
};

// Calculate Expense Volatility Score
const calculateExpenseVolatilityScore = (result) => {
  const monthlyTrends = result.chart_data?.monthly_trends || [];
  if (monthlyTrends.length < 2) return 0;
  const expenses = monthlyTrends.map(t => t.expenses);
  const mean = expenses.reduce((sum, val) => sum + val, 0) / expenses.length;
  const variance = expenses.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / expenses.length;
  const stdDev = Math.sqrt(variance);
  return Math.min(Math.max(Math.round((stdDev / mean) * 100), 0), 100);
};

// Calculate Loan Affordability Ratio
const calculateLoanAffordabilityRatio = (result) => {
  const monthlyIncome = result.financial_metrics?.monthly_income || 0;
  const monthlyExpenses = result.financial_metrics?.monthly_expenses || 0;
  const recommendedLoan = parseFloat(result.decision_summary?.recommended_loan_amount) || 0;
  const disposableIncome = monthlyIncome - monthlyExpenses;
  if (disposableIncome <= 0) return 0;
  const ratio = (recommendedLoan / 12) / disposableIncome; // Assuming 12-month loan term
  return Math.round(ratio * 100);
};

// Generate Insights Summary
const generateInsightsSummary = ({ incomeStabilityIndex, expenseVolatilityScore, loanAffordabilityRatio, cibilScore, riskCategory }) => {
  const insights = [];
  if (incomeStabilityIndex > 80) insights.push('High income stability indicates reliable cash flow.');
  else if (incomeStabilityIndex < 50) insights.push('Low income stability suggests repayment challenges.');
  if (expenseVolatilityScore > 50) insights.push('High expense volatility may impact planning.');
  else if (expenseVolatilityScore < 20) insights.push('Low expense volatility supports affordability.');
  if (loanAffordabilityRatio > 100) insights.push('Loan exceeds disposable income, indicating strain.');
  else if (loanAffordabilityRatio < 50) insights.push('Loan is within disposable income, suggesting affordability.');
  if (cibilScore >= 750) insights.push('Excellent CIBIL score enhances approval chances.');
  if (riskCategory === 'High Risk') insights.push('High risk profile requires mitigation strategies.');
  return insights;
};

// Save report to database
const saveReportToDatabase = async (reportData, userId, fileId) => {
  try {
    const { data: { session }, error: sessionError } = await supabase.auth.getSession();
    if (sessionError || !session) {
      throw new Error('No user session found. Please log in.');
    }
    const token = session.access_token;
    console.log('Saving report with token:', token);
    console.log('Request body:', {
      file_id: fileId,
      report_details: reportData,
      user_id: userId,
      action: reportData.decision_summary.final_decision,
      reason: reportData.decision_summary.reason,
      confidence: parseFloat(reportData.decision_summary.confidence_score),
    });

    const response = await fetch(API_ENDPOINTS.SAVE_REPORT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({
        file_id: fileId,
        report_details: reportData,
        user_id: userId,
        action: reportData.decision_summary.final_decision,
        reason: reportData.decision_summary.reason,
        confidence: parseFloat(reportData.decision_summary.confidence_score),
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error('Save report error response:', errorData);
      throw new Error(`Failed to save report: ${errorData.detail || response.statusText}`);
    }

    const savedReport = await response.json();
    console.log('Report saved:', savedReport);
    return savedReport;
  } catch (error) {
    console.error('Error saving report:', error);
    throw error;
  }
};

// Reports Component
const Reports = ({ backendUrl }) => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedReport, setSelectedReport] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    try {
      const { user } = await supabase.auth.getUser();
      const token = user?.access_token;
      const response = await fetch(API_ENDPOINTS.GET_REPORTS, {
        method: 'GET',
        credentials: 'include',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setReports(data);
      }
    } catch (err) {
      console.error('Error fetching reports:', err);
    } finally {
      setLoading(false);
    }
  };
};
const handleDownloadPDF = async (report, index) => {
    try {
      setSelectedReport(index);
      await new Promise(resolve => setTimeout(resolve, 500)); // Wait for render
      const refs = { monthlyTrendRef, categoryPieRef, riskFactorRef, transactionVolumeRef, financialRatiosRef };
      const areRefsReady = Object.values(refs).every(ref => ref.current && ref.current.offsetParent !== null);
      if (!areRefsReady) {
        throw new Error('Charts are not fully loaded');
      }
      const result = await generatePDFReport(report, refs);
      console.log('PDF generated:', result);
      alert(`PDF report generated: ${result.filename}`);
    } catch (error) {
      console.error('PDF generation failed:', error);
      alert(`Failed to generate PDF report: ${error.message}`);
    }
  };

  const deleteReport = async (reportId) => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) throw new Error('No user session found');
      const token = session.access_token;
      // Assuming a DELETE endpoint exists; adjust if necessary
      const response = await fetch(`${API_ENDPOINTS.GET_REPORTS}/${reportId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      if (!response.ok) throw new Error('Failed to delete report');
      setReports(prev => prev.filter(report => report.report_id !== reportId));
      if (selectedReport === reportId) setSelectedReport(null);
    } catch (error) {
      console.error('Delete report failed:', error);
      alert(`Failed to delete report: ${error.message}`);
    }
  };

// PDF Generation utility - Enhanced for professional corporate format
const generatePDFReport = async (analysisData, chartRefs) => {
  const pdf = new jsPDF('p', 'mm', 'a4');
  const pageWidth = 210;
  const pageHeight = 297;
  const margin = 10;
  let yOffset = margin;

  // Helper function to add header
  const addHeader = () => {
    pdf.setFillColor(33, 37, 41);
    pdf.rect(0, 0, pageWidth, 20, 'F');
    pdf.setTextColor(255, 255, 255);
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(14);
    pdf.text('Corporate Loan Analytics Report', margin, 13);
    pdf.setFontSize(10);
    pdf.setFont('helvetica', 'normal');
    pdf.text(`Generated: ${new Date().toLocaleDateString()}`, pageWidth - margin - 50, 13);
  };

  // Helper function to add footer with page number
  const addFooter = (pageNum) => {
    pdf.setTextColor(100);
    pdf.setFontSize(8);
    pdf.text(`Page ${pageNum} | &copy; 2025 FinSight. All rights reserved.`, pageWidth / 2, pageHeight - 5, { align: 'center' });
  };

  // Helper function to add section title
  const addSectionTitle = (title) => {
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(12);
    pdf.setTextColor(0);
    pdf.text(title.toUpperCase(), margin, yOffset);
    pdf.setLineWidth(0.5);
    pdf.line(margin, yOffset + 2, pageWidth - margin, yOffset + 2);
    yOffset += 10;
    if (yOffset > pageHeight - 30) {
      pdf.addPage();
      addHeader();
      yOffset = 25;
    }
  };

  // Helper function to add paragraph
  const addParagraph = (text, fontSize = 10) => {
    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(fontSize);
    const lines = pdf.splitTextToSize(text, pageWidth - 2 * margin);
    pdf.text(lines, margin, yOffset);
    yOffset += (lines.length * fontSize * 0.7) + 5;
    if (yOffset > pageHeight - 30) {
      pdf.addPage();
      addHeader();
      yOffset = 25;
    }
  };

  // Helper function to add table
  const addTable = (headers, rows, title = null) => {
    if (title) {
      addParagraph(title, 11);
    }
    if (yOffset + (rows.length * 10) > pageHeight - 30) {
      pdf.addPage();
      addHeader();
      yOffset = 25;
    }
    autoTable(pdf, {
      head: [headers],
      body: rows,
      startY: yOffset,
      margin: { left: margin, right: margin },
      theme: 'grid',
      styles: { fontSize: 9, cellPadding: 3, overflow: 'linebreak' },
      headStyles: { fillColor: [33, 37, 41], textColor: 255 },
      alternateRowStyles: { fillColor: [240, 240, 240] },
    });
    yOffset = pdf.lastAutoTable?.finalY ? pdf.lastAutoTable.finalY + 10 : yOffset + (rows.length * 10) + 10;
  };

  // Helper function to add image
  const addImage = async (element, title = null, maxHeight = 100) => {
    if (!element) {
      console.warn(`Chart element for ${title} is null, skipping`);
      addParagraph(`[Chart: ${title} could not be rendered]`, 10);
      return;
    }
    if (title) {
      addSectionTitle(title);
    }
    if (yOffset + maxHeight > pageHeight - 30) {
      pdf.addPage();
      addHeader();
      yOffset = 25;
    }
    try {
      const canvas = await html2canvas(element, {
        scale: 2,
        backgroundColor: '#FFFFFF',
        useCORS: true,
        logging: true,
      });
      const imgData = canvas.toDataURL('image/png');
      const imgProps = pdf.getImageProperties(imgData);
      const width = pageWidth - 2 * margin;
      const height = Math.min((imgProps.height * width) / imgProps.width, maxHeight);
      pdf.addImage(imgData, 'PNG', margin, yOffset, width, height);
      yOffset += height + 10;
    } catch (error) {
      console.error(`Failed to capture chart ${title}:`, error);
      addParagraph(`[Error rendering chart: ${title}]`, 10);
    }
  };

  try {
    // Validate analysisData
    if (!analysisData?.financial_metrics || !analysisData?.decision_summary || !analysisData?.risk_assessment) {
      throw new Error('Invalid analysis data structure');
    }

    // Add cover page
    pdf.setFontSize(20);
    pdf.setTextColor(0);
    pdf.text(`Loan Analysis Report`, pageWidth / 2, 80, { align: 'center' });
    pdf.setFontSize(14);
    pdf.text(analysisData.file_analysis?.applicant_name || 'Unknown Applicant', pageWidth / 2, 100, { align: 'center' });
    pdf.setFontSize(12);
    pdf.text(`CIBIL Score: ${analysisData.financial_metrics?.cibil_score || 'N/A'}`, pageWidth / 2, 120, { align: 'center' });
    pdf.text(`Final Decision: ${analysisData.decision_summary?.final_decision || 'N/A'}`, pageWidth / 2, 135, { align: 'center' });
    addFooter(1);

    // Page 2: Executive Summary
    pdf.addPage();
    addHeader();
    yOffset = 25;
    addSectionTitle('Executive Summary');
    addParagraph(`Applicant: ${analysisData.file_analysis?.applicant_name || 'N/A'}`);
    addParagraph(`File Analyzed: ${analysisData.file_analysis?.file_name || 'N/A'}`);
    addParagraph(`Final Decision: ${analysisData.decision_summary?.final_decision || 'N/A'}`);
    addParagraph(`Reason: ${analysisData.decision_summary?.reason || 'N/A'}`);
    addParagraph(`Confidence Score: ${analysisData.decision_summary?.confidence_score ? (parseFloat(analysisData.decision_summary.confidence_score) * 100).toFixed(1) + '%' : 'N/A'}`);
    addParagraph(`Recommended Loan Amount: ${analysisData.decision_summary?.recommended_loan_amount || 'N/A'}`);
    addParagraph(`Interest Rate Bracket: ${analysisData.decision_summary?.interest_rate_bracket || 'N/A'}`);
    addParagraph(`Risk Category: ${analysisData.risk_assessment?.risk_category || 'N/A'}`);

    // Key Financial Metrics Table
    addSectionTitle('Key Financial Metrics');
    const metricsTable = [
      ['Monthly Income', analysisData.financial_metrics?.monthly_income ? `₹${analysisData.financial_metrics.monthly_income.toLocaleString()}` : 'N/A'],
      ['Monthly Expenses', analysisData.financial_metrics?.monthly_expenses ? `₹${analysisData.financial_metrics.monthly_expenses.toLocaleString()}` : 'N/A'],
      ['CIBIL Score', analysisData.financial_metrics?.cibil_score || 'N/A'],
      ['Savings Rate', analysisData.financial_metrics?.savings_rate ? `${analysisData.financial_metrics.savings_rate}%` : 'N/A'],
      ['Debt-to-Income Ratio', analysisData.financial_metrics?.debt_to_income_ratio ? `${analysisData.financial_metrics.debt_to_income_ratio}%` : 'N/A'],
      ['Average Balance', analysisData.financial_metrics?.average_balance ? `₹${analysisData.financial_metrics.average_balance.toLocaleString()}` : 'N/A'],
      ['Total Transactions', analysisData.financial_metrics?.total_transactions || 'N/A'],
      ['Income Variability', analysisData.financial_metrics?.income_variability || 'N/A'],
      ['Income Stability Index', analysisData.additional_insights?.income_stability_index ? `${analysisData.additional_insights.income_stability_index}/100` : 'N/A'],
      ['Expense Volatility Score', analysisData.additional_insights?.expense_volatility_score ? `${analysisData.additional_insights.expense_volatility_score}/100` : 'N/A'],
      ['Loan Affordability Ratio', analysisData.additional_insights?.loan_affordability_ratio ? `${analysisData.additional_insights.loan_affordability_ratio}%` : 'N/A'],
    ];
    addTable(['Metric', 'Value'], metricsTable);

    // Risk Assessment Table
    addSectionTitle('Risk Assessment');
    addParagraph(`Overall Risk Score: ${analysisData.risk_assessment?.overall_risk_score || 'N/A'}/100`);
    addParagraph(`Risk Category: ${analysisData.risk_assessment?.risk_category || 'N/A'}`);
    const riskTable = (analysisData.risk_assessment?.risk_factors || []).map(f => [f.factor || 'N/A', f.score || 'N/A']);
    addTable(['Risk Factor', 'Score'], riskTable);

    // Compliance Checks Table
    addSectionTitle('Compliance and Security Verification');
    const complianceTable = [
      ['AML Status', analysisData.compliance_checks?.aml_status || 'N/A'],
      ['Fraud Indicators', analysisData.compliance_checks?.fraud_indicators ? `${analysisData.compliance_checks.fraud_indicators} detected` : 'N/A'],
    ];
    addTable(['Check', 'Status'], complianceTable);

    // Detailed Analysis Table
    addSectionTitle('Detailed Financial Analysis');
    const detailedTable = [
      ['Account Stability', analysisData.detailed_analysis?.account_stability || 'N/A'],
      ['Cash Flow Pattern', analysisData.detailed_analysis?.cash_flow_pattern || 'N/A'],
      ['Transaction Frequency', analysisData.detailed_analysis?.transaction_frequency || 'N/A'],
      ['Banking Behavior', analysisData.detailed_analysis?.banking_behavior || 'N/A'],
      ['Seasonal Variations', analysisData.detailed_analysis?.seasonal_variations || 'N/A'],
    ];
    addTable(['Indicator', 'Value'], detailedTable);

    // Additional Insights
    addSectionTitle('Additional Financial Insights');
    const insightsTable = [
      ['Income Stability Index', analysisData.additional_insights?.income_stability_index ? `${analysisData.additional_insights.income_stability_index}/100` : 'N/A'],
      ['Expense Volatility Score', analysisData.additional_insights?.expense_volatility_score ? `${analysisData.additional_insights.expense_volatility_score}/100` : 'N/A'],
      ['Loan Affordability Ratio', analysisData.additional_insights?.loan_affordability_ratio ? `${analysisData.additional_insights.loan_affordability_ratio}%` : 'N/A'],
    ];
    addTable(['Insight', 'Value'], insightsTable);
    addParagraph('Summary of Insights:');
    analysisData.additional_insights?.insights_summary.forEach(insight => addParagraph(`• ${insight}`));

    // Charts Section
    pdf.addPage();
    addHeader();
    yOffset = 25;
    addSectionTitle('Data Visualizations and Insights');

    await addImage(chartRefs.monthlyTrendRef.current, 'Monthly Income vs Expenses Trend');
    addParagraph('Insight: The trend shows consistent income with controlled expenses, indicating good financial stability.');

    await addImage(chartRefs.categoryPieRef.current, 'Category-wise Expense Breakdown');
    addParagraph('Insight: Housing and food dominate expenses, suggesting potential areas for cost optimization.');

    await addImage(chartRefs.riskFactorRef.current, 'Risk Factor Analysis (Radar Chart)');
    addParagraph('Insight: Low risk in income stability but higher in debt load, recommending debt reduction strategies.');

    await addImage(chartRefs.transactionVolumeRef.current, 'Transaction Volume Over Time');
    addParagraph('Insight: Steady transaction volume with no unusual spikes, indicating normal activity.');

    await addImage(chartRefs.financialRatiosRef.current, 'Key Financial Ratios');
    addParagraph('Insight: Balanced ratios indicate strong financial health.');

    // Recommendations
    pdf.addPage();
    addHeader();
    yOffset = 25;
    addSectionTitle('Recommendations and Insights');
    addParagraph('Based on the analysis, the following recommendations are provided to improve financial health or mitigate risks:');
    (analysisData.recommendations || []).forEach((rec, idx) => {
      addParagraph(`${idx + 1}. ${rec || 'N/A'}`);
    });

    // Add footers to all pages
    const pages = pdf.internal.getNumberOfPages();
    for (let i = 1; i <= pages; i++) {
      pdf.setPage(i);
      addFooter(i);
    }

    // Save PDF
    const filename = `loan_analysis_report_${analysisData.file_analysis?.applicant_name || 'unknown'}_${Date.now()}.pdf`;
    pdf.save(filename);

    return {
      success: true,
      filename,
      size: pdf.output('blob').size,
    };
  } catch (error) {
    console.error('PDF generation failed:', error);
    throw new Error(`Failed to generate PDF: ${error.message}`);
  }
};

// Chart Components
const MonthlyTrendChart = ({ data }) => (
  <ResponsiveContainer width="100%" height={300}>
    <LineChart data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="month" />
      <YAxis />
      <Tooltip formatter={(value) => [`₹${value.toLocaleString()}`, '']} />
      <Line type="monotone" dataKey="income" stroke="#8884d8" strokeWidth={3} name="Income" />
      <Line type="monotone" dataKey="expenses" stroke="#82ca9d" strokeWidth={3} name="Expenses" />
      <Line type="monotone" dataKey="balance" stroke="#ffc658" strokeWidth={3} name="Balance" />
    </LineChart>
  </ResponsiveContainer>
);

const CategoryPieChart = ({ data }) => (
  <ResponsiveContainer width="100%" height={300}>
    <PieChart>
      <Pie
        data={data}
        cx="50%"
        cy="50%"
        labelLine={false}
        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
        outerRadius={80}
        fill="#8884d8"
        dataKey="value"
      >
        {data.map((entry, index) => (
          <Cell key={`cell-${index}`} fill={entry.color} />
        ))}
      </Pie>
      <Tooltip formatter={(value) => [`₹${value.toLocaleString()}`, 'Amount']} />
    </PieChart>
  </ResponsiveContainer>
);

const RiskFactorChart = ({ data }) => (
  <ResponsiveContainer width="100%" height={300}>
    <RadarChart data={data}>
      <PolarGrid />
      <PolarAngleAxis dataKey="factor" />
      <Radar name="Risk Score" dataKey="score" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
      <Tooltip formatter={(value) => [`${value}/100`, 'Score']} />
    </RadarChart>
  </ResponsiveContainer>
);

const TransactionVolumeChart = ({ data }) => (
  <ResponsiveContainer width="100%" height={300}>
    <AreaChart data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="month" />
      <YAxis />
      <Tooltip formatter={(value, name) => [
        name === 'volume' ? value : `₹${value.toLocaleString()}`,
        name === 'volume' ? 'Transactions' : 'Amount'
      ]} />
      <Area type="monotone" dataKey="volume" stackId="1" stroke="#8884d8" fill="#8884d8" />
    </AreaChart>
  </ResponsiveContainer>
);

const FinancialRatiosChart = ({ data }) => (
  <ResponsiveContainer width="100%" height={300}>
    <BarChart data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="name" />
      <YAxis />
      <Tooltip formatter={(value) => [`${value}%`, 'Value']} />
      <Bar dataKey="value" fill="#8884d8" />
    </BarChart>
  </ResponsiveContainer>
);

const LoanAnalyticsSystem = () => {
  const [step, setStep] = useState(1);
  const [currentView, setCurrentView] = useState('upload');
  const [cibilScore, setCibilScore] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);
  const [user, setUser] = useState(null);
  const [savedReports, setSavedReports] = useState([]);

  // Chart refs for PDF capture
  const monthlyTrendRef = useRef(null);
  const categoryPieRef = useRef(null);
  const riskFactorRef = useRef(null);
  const transactionVolumeRef = useRef(null);
  const financialRatiosRef = useRef(null);

  useEffect(() => {
    const getUser = async () => {
      const { data: { user }, error } = await supabase.auth.getUser();
      if (error) {
        console.error('Error fetching user:', error);
        alert('Please log in to continue');
        return;
      }
      setUser(user);
    };
    getUser();
    
    const verifyAPI = async () => {
      const isHealthy = await checkAPIHealth();
      if (!isHealthy) {
        console.error('API health check failed. Please ensure the backend server is running.');
        alert('Cannot connect to the analysis server. Please try again later.');
      } else {
        console.log('API health check passed');
      }
    };
    verifyAPI();
  }, []);

  const handleCibilSubmit = () => {
    const score = parseInt(cibilScore);
    if (score >= 300 && score <= 900) {
      setStep(2);
    } else {
      alert('Please enter a valid CIBIL score between 300 and 900.');
    }
  };

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files || []).filter((file) =>
      ['application/pdf', 'text/csv', 'application/vnd.ms-excel', 
       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
       'text/plain'].includes(file.type)
    );
    setUploadedFiles(files);
  };

  const handleAnalysis = async () => {
    if (!user) {
      alert('Please log in to analyze documents');
      return;
    }
    if (uploadedFiles.length === 0) {
      alert("Please upload at least one file");
      return;
    }
    if (!cibilScore || isNaN(cibilScore) || cibilScore < 300 || cibilScore > 900) {
      alert('Please enter a valid CIBIL score between 300 and 900');
      return;
    }
    setIsAnalyzing(true);
    try {
      const results = await Promise.all(
        uploadedFiles.map(async (file) => {
        const analysisResult = await analyzeFile(file, parseInt(cibilScore));
        console.log('File ID from analysis:', analysisResult.file_analysis?.file_id);
        if (!analysisResult.file_analysis?.file_id) {
          throw new Error('File ID is missing in analysis result');
        }
        // Save to database
        const savedReport = await saveReportToDatabase(
          analysisResult,
          user.id,
          analysisResult.file_analysis.file_id
        );
        return { ...analysisResult, report_id: savedReport.report_id };
      })
      );
      console.log('All analysis results:', results);
      setAnalysisResults(results);
      setSavedReports(prev => [...prev, ...results]);
      setStep(3);
    } catch (error) {
      console.error("Analysis failed:", error);
      alert(`Analysis failed: ${error.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDownloadPDF = async (analysisData, index) => {
    try {
      setSelectedResult(index); // Ensure charts are visible
      await new Promise(resolve => setTimeout(resolve, 500)); // Wait for render
      const refs = [monthlyTrendRef, categoryPieRef, riskFactorRef, transactionVolumeRef];
      const areRefsReady = refs.every(ref => ref.current && ref.current.offsetParent !== null);
      if (!areRefsReady) {
        throw new Error('Charts are not fully loaded');
      }
      const result = await generatePDFReport(analysisData, {
        monthlyTrendRef,
        categoryPieRef,
        riskFactorRef,
        transactionVolumeRef,
        financialRatiosRef,
      });
      console.log('PDF generated:', result);
      alert(`PDF report generated: ${result.filename}`);
    } catch (error) {
      console.error('PDF generation failed:', error);
      alert(`Failed to generate PDF report: ${error.message}`);
    }
  };

  const resetSystem = () => {
    setStep(1);
    setCibilScore('');
    setUploadedFiles([]);
    setAnalysisResults([]);
    setSelectedResult(null);
  };

  const deleteReport = (reportId) => {
    setSavedReports(prev => prev.filter(report => report.id !== reportId));
    if (selectedResult === reportId) {
      setSelectedResult(null);
    }
  };

  const getRiskColor = (riskCategory) => {
    return RISK_COLORS[riskCategory] || '#6B7280';
  };

  const getDecisionColor = (decision) => {
    return decision === 'APPROVED' 
      ? 'text-green-600 bg-green-50 border-green-200' 
      : 'text-red-600 bg-red-50 border-red-200';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex space-x-4">
              <button
                onClick={() => setCurrentView('upload')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  currentView === 'upload' 
                    ? 'bg-blue-600 text-white' 
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                New Analysis
              </button>
              <button
                onClick={() => setCurrentView('reports')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  currentView === 'reports'
                    ? 'bg-purple-600 text-white' 
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <FileText className="h-4 w-4 mr-2 inline" />
                Reports ({savedReports.length})
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-6">
        {/* Navigation Tabs */}
        {currentView === 'reports' && (
          <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6 mb-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">Analysis Reports</h2>
              <button
                onClick={() => setCurrentView('upload')}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium transition-colors"
              >
                New Analysis
              </button>
            </div>

            {savedReports.length === 0 ? (
              <div className="text-center py-12">
                <FileText className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 text-lg">No reports generated yet</p>
                <p className="text-gray-500">Upload and analyze documents to generate reports</p>
              </div>
            ) : (
              <div className="space-y-4">
                {savedReports.map((report, index) => (
                  <div key={report.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-4">
                          <div>
                            <h3 className="font-semibold text-gray-900">{report.file_analysis.applicant_name || 'Unknown Applicant'}</h3>
                            <p className="text-sm text-gray-600">
                              {report.file_analysis.file_name} • Generated {new Date(report.generated_at).toLocaleDateString()}
                            </p>
                          </div>
                          <div className={`px-3 py-1 rounded-full text-sm font-medium ${getDecisionColor(report.decision_summary.final_decision)}`}>
                            {report.decision_summary.final_decision}
                          </div>
                          <div className="text-right">
                            <div className="text-lg font-bold text-blue-600">
                              CIBIL: {report.financial_metrics.cibil_score}
                            </div>
                            <div className={`text-sm font-medium`} style={{color: getRiskColor(report.risk_assessment.risk_category)}}>
                              {report.risk_assessment.risk_category}
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2 ml-4">
                        <button
                          onClick={() => setSelectedResult(selectedResult === report.id ? null : report.id)}
                          className="p-2 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                          title="View Details"
                        >
                          <Eye className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => handleDownloadPDF(report, index)}
                          className="p-2 text-gray-600 hover:text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                          title="Download PDF"
                        >
                          <FileDown className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => deleteReport(report.id)}
                          className="p-2 text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                          title="Delete Report"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>

                    {selectedResult === report.id && (
                      <div className="mt-6 border-t pt-6">
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                          <div className="space-y-4">
                            <h4 className="font-semibold text-gray-900 mb-3">Key Metrics</h4>
                            <div className="grid grid-cols-2 gap-3">
                              <div className="bg-blue-50 p-3 rounded-lg">
                                <div className="text-2xl font-bold text-blue-600">
                                  ₹{report.financial_metrics.monthly_income.toLocaleString()}
                                </div>
                                <div className="text-sm text-blue-700">Monthly Income</div>
                              </div>
                              <div className="bg-green-50 p-3 rounded-lg">
                                <div className="text-2xl font-bold text-green-600">
                                  {report.financial_metrics.savings_rate}%
                                </div>
                                <div className="text-sm text-green-700">Savings Rate</div>
                              </div>
                              <div className="bg-purple-50 p-3 rounded-lg">
                                <div className="text-2xl font-bold text-purple-600">
                                  {report.financial_metrics.debt_to_income_ratio}%
                                </div>
                                <div className="text-sm text-purple-700">DTI Ratio</div>
                              </div>
                              <div className="bg-orange-50 p-3 rounded-lg">
                                <div className="text-2xl font-bold text-orange-600">
                                  {report.risk_assessment.overall_risk_score}
                                </div>
                                <div className="text-sm text-orange-700">Risk Score</div>
                              </div>
                            </div>
                          </div>

                          <div className="space-y-4">
                            <h4 className="font-semibold text-gray-900 mb-3">Decision Analysis</h4>
                            <div className="bg-gray-50 p-4 rounded-lg">
                              <p className="text-sm text-gray-700 mb-2">
                                <strong>Reason:</strong> {report.decision_summary.reason}
                              </p>
                              <p className="text-sm text-gray-700 mb-2">
                                <strong>Confidence:</strong> {(parseFloat(report.decision_summary.confidence_score) * 100).toFixed(1)}%
                              </p>
                              <p className="text-sm text-gray-700">
                                <strong>Recommended Amount:</strong> {report.decision_summary.recommended_loan_amount}
                              </p>
                            </div>
                          </div>
                        </div>

                        <div className="mt-6">
                          <h4 className="font-semibold text-gray-900 mb-4">Financial Analysis Charts</h4>
                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <div className="bg-white border rounded-lg p-4" ref={monthlyTrendRef}>
                              <h5 className="font-medium text-gray-900 mb-3">Monthly Trends</h5>
                              <MonthlyTrendChart data={report.chart_data.monthly_trends} />
                            </div>
                            <div className="bg-white border rounded-lg p-4" ref={categoryPieRef}>
                              <h5 className="font-medium text-gray-900 mb-3">Category Breakdown</h5>
                              <CategoryPieChart data={report.chart_data.category_breakdown} />
                            </div>
                            <div className="bg-white border rounded-lg p-4" ref={riskFactorRef}>
                              <h5 className="font-medium text-gray-900 mb-3">Risk Factors</h5>
                              <RiskFactorChart data={report.risk_assessment.risk_factors} />
                            </div>
                            <div className="bg-white border rounded-lg p-4" ref={transactionVolumeRef}>
                              <h5 className="font-medium text-gray-900 mb-3">Transaction Volume</h5>
                              <TransactionVolumeChart data={report.chart_data.transaction_volume} />
                            </div>
                          </div>
                        </div>

                        <div className="mt-6">
                          <h4 className="font-semibold text-gray-900 mb-3">Recommendations</h4>
                          <div className="bg-blue-50 rounded-lg p-4">
                            <ul className="space-y-2">
                              {report.recommendations.map((rec, idx) => (
                                <li key={idx} className="flex items-start">
                                  <CheckCircle className="h-4 w-4 text-blue-600 mr-2 mt-0.5 flex-shrink-0" />
                                  <span className="text-sm text-blue-800">{rec}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {currentView === 'upload' && (
          <>
            <div className="mb-8 flex items-center justify-center space-x-8">
              {[1, 2, 3].map((s) => {
                const isActive = step >= s;
                const isComplete = step > s;
                const Icon = s === 1 ? CreditCard : s === 2 ? Upload : CheckCircle;
                const label = s === 1 ? 'CIBIL Score' : s === 2 ? 'Upload & Analyze' : 'Results';
                
                return (
                  <div key={s} className="flex items-center">
                    <div className={`flex items-center ${isActive ? 'text-blue-600' : 'text-gray-400'}`}>
                      <div className={`w-12 h-12 rounded-full flex items-center justify-center border-2 ${
                        isComplete ? 'bg-green-600 border-green-600 text-white' :
                        isActive ? 'bg-blue-600 border-blue-600 text-white' : 'bg-white border-gray-300'
                      }`}>
                        <Icon className="h-6 w-6" />
                      </div>
                      <div className="ml-3">
                        <div className="font-medium">{label}</div>
                        <div className="text-sm text-gray-500">
                          {s === 1 ? 'Enter credit score' : s === 2 ? 'Upload documents' : 'View analysis'}
                        </div>
                      </div>
                    </div>
                    {s < 3 && (
                      <div className={`h-px w-16 mx-4 ${step > s ? 'bg-green-600' : 'bg-gray-300'}`}></div>
                    )}
                  </div>
                );
              })}
            </div>

            {step === 1 && (
              <div className="max-w-md mx-auto">
                <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8">
                  <div className="text-center mb-8">
                    <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
                      <CreditCard className="h-10 w-10 text-blue-600" />
                    </div>
                    <h2 className="text-2xl font-bold text-gray-900 mb-3">Enter Your CIBIL Score</h2>
                    <p className="text-gray-600">Your credit score helps us provide accurate loan analysis</p>
                  </div>

                  <div className="space-y-6">
                    <div>
                      <input
                        type="number"
                        min="300"
                        max="900"
                        value={cibilScore}
                        onChange={(e) => setCibilScore(e.target.value)}
                        className="w-full px-6 py-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent text-center text-3xl font-bold text-gray-900"
                        placeholder="750"
                      />
                      <div className="flex justify-between text-sm text-gray-500 mt-2 px-2">
                        <span>Min: 300</span>
                        <span>Max: 900</span>
                      </div>
                    </div>

                    <div className="bg-blue-50 rounded-lg p-4">
                      <h4 className="font-medium text-blue-900 mb-2">CIBIL Score Ranges:</h4>
                      <div className="text-sm text-blue-800 space-y-1">
                        <div>• 750-900: Excellent (Best rates)</div>
                        <div>• 650-749: Good (Competitive rates)</div>
                        <div>• 550-649: Fair (Higher rates)</div>
                        <div>• 300-549: Poor (Limited options)</div>
                      </div>
                    </div>

                    <button
                      onClick={handleCibilSubmit}
                      disabled={!cibilScore || parseInt(cibilScore) < 300 || parseInt(cibilScore) > 900}
                      className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white py-4 px-6 rounded-xl font-medium transition-colors text-lg"
                    >
                      Continue to Upload
                    </button>
                  </div>
                </div>
              </div>
            )}

            {step === 2 && (
              <div className="max-w-2xl mx-auto">
                <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8">
                  <div className="text-center mb-8">
                    <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
                      <Upload className="h-10 w-10 text-blue-600" />
                    </div>
                    <h2 className="text-2xl font-bold text-gray-900 mb-3">Upload Financial Documents</h2>
                    <p className="text-gray-600">Upload bank statements, salary slips, or financial documents</p>
                    <div className="mt-2 text-sm text-blue-600 bg-blue-50 rounded-lg p-2">
                      CIBIL Score: <strong>{cibilScore}</strong>
                    </div>
                  </div>

                  <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center mb-6 hover:border-blue-400 transition-colors bg-gray-50">
                    <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <label className="cursor-pointer">
                      <span className="text-blue-600 hover:text-blue-700 font-medium text-lg">Click to upload files</span>
                      <p className="text-sm text-gray-500 mt-2">Supports PDF, CSV, Excel, and Text files</p>
                      <p className="text-xs text-gray-400 mt-1">Maximum file size: 50MB each</p>
                      <input
                        type="file"
                        multiple
                        accept=".pdf,.csv,.xlsx,.xls,.txt"
                        onChange={handleFileUpload}
                        className="hidden"
                      />
                    </label>
                  </div>

                  {uploadedFiles.length > 0 && (
                    <div className="mb-6">
                      <h3 className="font-medium text-gray-900 mb-4">Uploaded Files ({uploadedFiles.length}):</h3>
                      <div className="space-y-3 max-h-48 overflow-y-auto">
                        {uploadedFiles.map((file, idx) => (
                          <div key={idx} className="flex items-center p-4 bg-blue-50 rounded-lg border border-blue-100">
                            <FileText className="h-5 w-5 text-blue-600 mr-3 flex-shrink-0" />
                            <div className="flex-1 min-w-0">
                              <div className="text-sm font-medium text-gray-900 truncate">{file.name}</div>
                              <div className="text-xs text-gray-500 mt-1">
                                {(file.size / 1024 / 1024).toFixed(2)} MB • {file.type.split('/')[1]?.toUpperCase()}
                              </div>
                            </div>
                            <CheckCircle className="h-5 w-5 text-green-600 ml-2" />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex gap-4">
                    <button
                      onClick={() => setStep(1)}
                      className="flex-1 border border-gray-300 hover:bg-gray-50 text-gray-700 py-3 px-6 rounded-xl font-medium transition-colors"
                    >
                      Back
                    </button>
                    <button
                      onClick={handleAnalysis}
                      disabled={uploadedFiles.length === 0 || isAnalyzing}
                      className="flex-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white py-3 px-8 rounded-xl font-medium transition-colors"
                    >
                      {isAnalyzing ? 'Analyzing Documents...' : 'Start Comprehensive Analysis'}
                    </button>
                  </div>

                  {isAnalyzing && (
                    <div className="mt-6 bg-blue-50 rounded-lg p-4">
                      <div className="flex items-center">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-3"></div>
                        <span className="text-blue-800 font-medium">Processing your documents...</span>
                      </div>
                      <div className="mt-2 text-sm text-blue-600">
                        Analyzing financial patterns, risk factors, and generating comprehensive reports...
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {step === 3 && analysisResults.length > 0 && (
              <div className="space-y-6">
                <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
                  <div className="text-center">
                    <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <CheckCircle className="h-8 w-8 text-green-600" />
                    </div>
                    <h2 className="text-2xl font-bold text-gray-900 mb-2">Comprehensive Analysis Complete</h2>
                    <p className="text-gray-600">Detailed loan eligibility analysis with charts and recommendations</p>
                  </div>
                </div>

                {analysisResults.map((result, index) => (
                  <div key={index} className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden">
                    <div className="bg-gradient-to-r from-blue-50 to-purple-50 px-6 py-4 border-b border-gray-200">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-lg font-semibold text-gray-900">
                            Analysis Report - {result.file_analysis.applicant_name || 'Unknown Applicant'}
                          </h3>
                          <p className="text-sm text-gray-600">{result.file_analysis.file_name}</p>
                        </div>
                        <div className="flex items-center space-x-4">
                          <div className={`px-4 py-2 rounded-full font-medium border ${getDecisionColor(result.decision_summary.final_decision)}`}>
                            {result.decision_summary.final_decision}
                          </div>
                          <button
                            onClick={() => handleDownloadPDF(result, index)}
                            className="flex items-center px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors"
                          >
                            <Download className="h-4 w-4 mr-2" />
                            PDF Report
                          </button>
                        </div>
                      </div>
                    </div>

                    <div className="p-6">
                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
                        <div className="bg-blue-50 p-4 rounded-lg text-center border border-blue-100">
                          <DollarSign className="h-8 w-8 text-blue-600 mx-auto mb-2" />
                          <div className="text-2xl font-bold text-blue-600">
                            {result.financial_metrics.cibil_score}
                          </div>
                          <div className="text-sm text-blue-700">CIBIL Score</div>
                        </div>
                        <div className="bg-green-50 p-4 rounded-lg text-center border border-green-100">
                          <TrendingUp className="h-8 w-8 text-green-600 mx-auto mb-2" />
                          <div className="text-2xl font-bold text-green-600">
                            ₹{result.financial_metrics.monthly_income.toLocaleString()}
                          </div>
                          <div className="text-sm text-green-700">Monthly Income</div>
                        </div>
                        <div className="bg-purple-50 p-4 rounded-lg text-center border border-purple-100">
                          <Target className="h-8 w-8 text-purple-600 mx-auto mb-2" />
                          <div className="text-2xl font-bold text-purple-600">
                            {result.financial_metrics.savings_rate}%
                          </div>
                          <div className="text-sm text-purple-700">Savings Rate</div>
                        </div>
                        <div className="p-4 rounded-lg text-center border" style={{
                          backgroundColor: `${getRiskColor(result.risk_assessment.risk_category)}15`,
                          borderColor: `${getRiskColor(result.risk_assessment.risk_category)}40`
                        }}>
                          <Shield className="h-8 w-8 mx-auto mb-2" style={{color: getRiskColor(result.risk_assessment.risk_category)}} />
                          <div className="text-lg font-bold" style={{color: getRiskColor(result.risk_assessment.risk_category)}}>
                            {result.risk_assessment.risk_category}
                          </div>
                          <div className="text-sm" style={{color: `${getRiskColor(result.risk_assessment.risk_category)}CC`}}>Risk Level</div>
                        </div>
                      </div>

                      <div className="mb-8">
                        <h4 className="font-semibold text-gray-900 mb-6 flex items-center">
                          <TrendingUp className="h-5 w-5 mr-2" />
                          Financial Analysis Charts
                        </h4>
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                          <div className="bg-gray-50 border rounded-lg p-4" ref={monthlyTrendRef}>
                            <h5 className="font-medium text-gray-900 mb-3">Monthly Income vs Expenses Trend</h5>
                            <MonthlyTrendChart data={result.chart_data.monthly_trends} />
                          </div>
                          <div className="bg-gray-50 border rounded-lg p-4" ref={categoryPieRef}>
                            <h5 className="font-medium text-gray-900 mb-3">Expense Category Breakdown</h5>
                            <CategoryPieChart data={result.chart_data.category_breakdown} />
                          </div>
                          <div className="bg-gray-50 border rounded-lg p-4" ref={riskFactorRef}>
                            <h5 className="font-medium text-gray-900 mb-3">Risk Factor Analysis</h5>
                            <RiskFactorChart data={result.risk_assessment.risk_factors} />
                          </div>
                          <div className="bg-gray-50 border rounded-lg p-4" ref={transactionVolumeRef}>
                            <h5 className="font-medium text-gray-900 mb-3">Transaction Volume Trend</h5>
                            <TransactionVolumeChart data={result.chart_data.transaction_volume} />
                          </div>
                        </div>
                      </div>

                      <div className="bg-gray-50 rounded-lg p-6 mb-6">
                        <h4 className="font-semibold text-gray-900 mb-4 flex items-center">
                          <Target className="h-5 w-5 mr-2" />
                          Comprehensive Decision Analysis
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="space-y-3">
                            <div className="flex justify-between">
                              <span className="text-gray-600">Decision:</span>
                              <span className={`font-medium ${result.decision_summary.final_decision === 'APPROVED' ? 'text-green-600' : 'text-red-600'}`}>
                                {result.decision_summary.final_decision}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Confidence:</span>
                              <span className="font-medium">{(parseFloat(result.decision_summary.confidence_score) * 100).toFixed(1)}%</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Risk Score:</span>
                              <span className="font-medium">{result.risk_assessment.overall_risk_score}/100</span>
                            </div>
                          </div>
                          <div className="space-y-3">
                            <div className="flex justify-between">
                              <span className="text-gray-600">Recommended Amount:</span>
                              <span className="font-medium text-blue-600">{result.decision_summary.recommended_loan_amount}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Interest Rate:</span>
                              <span className="font-medium">{result.decision_summary.interest_rate_bracket}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Processing Time:</span>
                              <span className="font-medium">{result.file_analysis.processing_time}</span>
                            </div>
                          </div>
                        </div>
                        <div className="mt-4 p-3 bg-white rounded border-l-4 border-blue-400">
                          <p className="text-sm text-gray-700">
                            <strong>Analysis Summary:</strong> {result.decision_summary.reason}
                          </p>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                        <div>
                          <h4 className="font-semibold text-gray-900 mb-3">Financial Health Indicators</h4>
                          <div className="space-y-2 text-sm bg-white border rounded-lg p-4">
                            <div className="flex justify-between">
                              <span className="text-gray-600">Average Balance:</span>
                              <span>₹{result.financial_metrics.average_balance.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Monthly Expenses:</span>
                              <span>₹{result.financial_metrics.monthly_expenses.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Debt-to-Income:</span>
                              <span>{result.financial_metrics.debt_to_income_ratio}%</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Total Transactions:</span>
                              <span>{result.financial_metrics.total_transactions}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Income Variability:</span>
                              <span>{result.financial_metrics.income_variability}</span>
                            </div>
                          </div>
                        </div>
                        <div>
                          <h4 className="font-semibold text-gray-900 mb-3">Banking Behavior Analysis</h4>
                          <div className="space-y-2 text-sm bg-white border rounded-lg p-4">
                            <div className="flex justify-between">
                              <span className="text-gray-600">Account Stability:</span>
                              <span className={result.detailed_analysis.account_stability === 'Stable' ? 'text-green-600' : 'text-orange-600'}>
                                {result.detailed_analysis.account_stability}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Cash Flow:</span>
                              <span className={result.detailed_analysis.cash_flow_pattern === 'Positive' ? 'text-green-600' : 'text-red-600'}>
                                {result.detailed_analysis.cash_flow_pattern}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Transaction Frequency:</span>
                              <span>{result.detailed_analysis.transaction_frequency}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Banking Behavior:</span>
                              <span>{result.detailed_analysis.banking_behavior}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Seasonal Impact:</span>
                              <span>{result.detailed_analysis.seasonal_variations}</span>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="mb-6">
                        <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
                          <Shield className="h-5 w-5 mr-2" />
                          Compliance & Security Verification
                        </h4>
                        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="space-y-2 text-sm">
                              <div className="flex justify-between">
                                <span className="text-gray-600">AML Status:</span>
                                <span className="text-green-600 font-medium">{result.compliance_checks.aml_status}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600">Document Authenticity:</span>
                                <span className="text-green-600 font-medium">{result.compliance_checks.document_authenticity}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600">Identity Verification:</span>
                                <span className="text-green-600 font-medium">{result.compliance_checks.identity_verification}</span>
                              </div>
                            </div>
                            <div className="space-y-2 text-sm">
                              <div className="flex justify-between">
                                <span className="text-gray-600">Address Verification:</span>
                                <span className="text-green-600 font-medium">{result.compliance_checks.address_verification}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600">Fraud Indicators:</span>
                                <span className={result.compliance_checks.fraud_indicators === 0 ? 'text-green-600' : 'text-red-600'}>
                                  {result.compliance_checks.fraud_indicators} detected
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="mb-6">
                        <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
                          <TrendingUp className="h-5 w-5 mr-2" />
                          Expert Recommendations
                        </h4>
                        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                          <ul className="space-y-2">
                            {result.recommendations.map((rec, idx) => (
                              <li key={idx} className="flex items-start">
                                <CheckCircle className="h-4 w-4 text-blue-600 mr-2 mt-0.5 flex-shrink-0" />
                                <span className="text-sm text-blue-800">{rec}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>

                      <div className="flex gap-3 pt-4 border-t">
                        <button
                          onClick={() => setSelectedResult(selectedResult === index ? null : index)}
                          className="flex-1 border border-gray-300 hover:bg-gray-50 text-gray-700 py-2 px-4 rounded-lg font-medium transition-colors"
                        >
                          {selectedResult === index ? 'Hide Technical Details' : 'View Technical Details'}
                        </button>
                        <button
                          onClick={() => handleDownloadPDF(result, index)}
                          className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
                        >
                          <Download className="h-4 w-4 mr-2" />
                          Download Full Report
                        </button>
                      </div>

                      {selectedResult === index && (
                        <div className="mt-6 p-4 bg-gray-50 rounded-lg border-t">
                          <h4 className="font-semibold text-gray-900 mb-3">Complete Technical Analysis Data</h4>
                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <div>
                              <h5 className="font-medium text-gray-800 mb-2">Raw Financial Metrics</h5>
                              <pre className="text-xs text-gray-700 overflow-x-auto whitespace-pre-wrap bg-white p-3 rounded border max-h-48 overflow-y-auto">
                                {JSON.stringify(result.financial_metrics, null, 2)}
                              </pre>
                            </div>
                            <div>
                              <h5 className="font-medium text-gray-800 mb-2">Risk Assessment Details</h5>
                              <pre className="text-xs text-gray-700 overflow-x-auto whitespace-pre-wrap bg-white p-3 rounded border max-h-48 overflow-y-auto">
                                {JSON.stringify(result.risk_assessment, null, 2)}
                              </pre>
                            </div>
                          </div>
                          <div className="mt-4">
                            <h5 className="font-medium text-gray-800 mb-2">Complete Analysis Metadata</h5>
                            <pre className="text-xs text-gray-700 overflow-x-auto whitespace-pre-wrap bg-white p-3 rounded border max-h-32 overflow-y-auto">
                              {JSON.stringify({
                                file_analysis: result.file_analysis,
                                generated_at: result.generated_at,
                                analysis_version: result.analysis_version
                              }, null, 2)}
                            </pre>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                <div className="text-center">
                  <button
                    onClick={resetSystem}
                    className="bg-gray-600 hover:bg-gray-700 text-white py-3 px-8 rounded-xl font-medium transition-colors"
                  >
                    Analyze New Documents
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default LoanAnalyticsSystem;