import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import supabase
from datetime import datetime
from pathlib import Path
import tempfile
import logging
import json

# Import from visualizer
from utils.visualizer import analyze_file
from cryptography.fernet import Fernet

# Setup logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Supabase client
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]  # Anon/public key
supabase = supabase.create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

#db
from services.files import save_uploaded_file, fetch_files
from services.reports import save_report
from services.base import client, admin_client

# Define base directory
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"

# Ensure directories exist
for directory in [OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# Load Fernet key from secrets
try:
    FERNET_KEY = st.secrets["FERNET_KEY"].encode()
    cipher_suite = Fernet(FERNET_KEY)
except KeyError:
    logger.error("FERNET_KEY not found in st.secrets. Encryption disabled.")
    st.error("Encryption key missing. Please configure FERNET_KEY in secrets to enable encryption.")
    cipher_suite = None

def extract_decision_action(decision):
    """Extract simple action from complex decision object"""
    if isinstance(decision, dict):
        action = decision.get('Action', '').lower().strip()
        if not action:
            action = decision.get('decision', '').lower().strip()
        if not action:
            action = decision.get('status', '').lower().strip()
        
        if 'approve' in action or 'accept' in action:
            return 'approve'
        elif 'reject' in action or 'deny' in action or 'decline' in action:
            return 'reject'
        else:
            return 'manual review'
    elif isinstance(decision, str):
        action = decision.lower().strip()
        if 'approve' in action or 'accept' in action:
            return 'approve'
        elif 'reject' in action or 'deny' in action or 'decline' in action:
            return 'reject'
        else:
            return 'manual review'
    else:
        return 'manual review'

def save_report_to_database(user_id: str, file_id: str, report_path: str, decision: dict, plots_paths: list = None):
    """Save the generated PDF report to database and storage - FIXED VERSION"""
    try:
        if not report_path or not os.path.exists(report_path):
            logger.error(f"Report file not found: {report_path}")
            return None
            
        # Validate inputs
        if not user_id:
            raise ValueError("user_id cannot be empty")
        if not file_id:
            raise ValueError("file_id cannot be empty")
        if not decision:
            raise ValueError("decision cannot be empty")
            
        # Convert file_id to int if it's a string
        try:
            file_id_int = int(file_id)
        except ValueError:
            raise ValueError(f"Invalid file_id: {file_id}")
        
        # Extract simple decision action
        simple_decision = extract_decision_action(decision)
        
        # Create simplified decision object for database
        db_decision = {
            "Action": simple_decision,
            "Reason": decision.get('Reason', 'Auto-generated decision') if isinstance(decision, dict) else 'Decision extracted'
        }
        
        # Read report content
        with open(report_path, 'rb') as f:
            report_content = f.read()
        
        # Upload PDF to Supabase storage
        report_filename = os.path.basename(report_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        storage_path = f"user_{user_id}/reports/{timestamp}_{report_filename}"
        
        logger.info(f"Uploading report to storage: {storage_path}")
        
        # Upload to storage using admin_client
        upload_response = admin_client.storage.from_('files').upload(
            storage_path, 
            report_content, 
            {
                'content-type': 'application/pdf', 
                'upsert': 'true'
            }
        )
        
        if hasattr(upload_response, 'error') and upload_response.error:
            raise ValueError(f"Storage upload failed: {upload_response.error}")
        
        # Save report metadata to database using the service
        report_id = save_report(
            user_id=user_id,
            file_id=file_id_int,
            decision=db_decision,  # Use simplified decision
            report_path=storage_path,
            plots_paths=plots_paths or []
        )
        
        if report_id:
            logger.info(f"Report saved successfully with ID: {report_id}, Decision: {simple_decision}")
            return report_id
        else:
            logger.error("Failed to save report to database")
            return None
            
    except Exception as e:
        logger.error(f"Error saving report to database: {str(e)}")
        return None
    
@st.cache_data
def process_file(file_content, file_type, file_name, models=None):
    result = {
        "raw_csv": None,
        "categorized_csv": None,
        "metrics_csv": None,
        "plots": [],
        "pdf_report": None,
        "decision_json": None,
        "metrics_df": pd.DataFrame(),
        "transactions_df": pd.DataFrame()
    }
    error = None
    tmp_file_path = None
    try:
        # Create a temporary file to store the uploaded content
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}")
        tmp_file_path = tmp_file.name
        tmp_file.write(file_content)
        tmp_file.flush()  # Ensure file is written
        tmp_file.close()  # Close the file to allow other processes to access it
        logger.info(f"[DEBUG] Temporary file created: {tmp_file_path}, exists: {os.path.exists(tmp_file_path)}")

        # Verify file accessibility
        if not os.path.exists(tmp_file_path):
            error = f"Temporary file creation failed: {tmp_file_path}"
            logger.error(error)
            return result, error, None, "Unknown", "Unknown"

        # Run visualizer pipeline
        with st.spinner("Generating full analysis and PDF report..."):
            results = analyze_file(tmp_file_path, cibil_score=720, fill_method="interpolate", out_dir=str(OUTPUT_DIR), applicant_data=None)

        # Display results
        if results:
            st.header("Loan Analysis Results")
            metrics_df = results.get("metrics_df", pd.DataFrame())
            monthly_income = float(metrics_df["Average Monthly Income"].iloc[0]) if not metrics_df.empty else 0.0
            monthly_expenses = float(metrics_df["Average Monthly Expenses"].iloc[0]) if not metrics_df.empty and "Average Monthly Expenses" in metrics_df.columns else 0.0
            net_surplus = float(metrics_df["Net Surplus"].iloc[0]) if not metrics_df.empty and "Net Surplus" in metrics_df.columns else monthly_income - monthly_expenses
            dti_ratio = float(metrics_df["DTI Ratio"].iloc[0]) if not metrics_df.empty and "DTI Ratio" in metrics_df.columns else 0.0
            red_flag_count = int(metrics_df["Red Flag Count"].iloc[0]) if not metrics_df.empty and "Red Flag Count" in metrics_df.columns else 0
            discretionary_spending = float(metrics_df["Discretionary Spending (%)"].iloc[0]) if "Discretionary Spending (%)" in metrics_df.columns else 0.0
            avg_monthly_balance = float(metrics_df["Average Closing Balance"].iloc[0]) if "Average Closing Balance" in metrics_df.columns else 0.0
            cash_withdrawals = float(metrics_df["Cash Withdrawals"].iloc[0]) if "Cash Withdrawals" in metrics_df.columns else 0.0
            existing_loans = int(metrics_df["Existing Loan Count"].iloc[0]) if "Existing Loan Count" in metrics_df.columns else 0
            bounced_cheques = int(metrics_df["Bounced Cheques Count"].iloc[0]) if "Bounced Cheques Count" in metrics_df.columns else 0
            overdraft_frequency = int(metrics_df["Overdraft Usage Frequency"].iloc[0]) if "Overdraft Usage Frequency" in metrics_df.columns else 0
            negative_balance_days = int(metrics_df["Negative Balance Days"].iloc[0]) if "Negative Balance Days" in metrics_df.columns else 0
            high_value_credits = int(metrics_df["Sudden High-Value Credits"].iloc[0]) if "Sudden High-Value Credits" in metrics_df.columns else 0
            circular_transactions = int(metrics_df["Circular Transactions"].iloc[0]) if "Circular Transactions" in metrics_df.columns else 0
            income_variability = float(metrics_df["Income Variability Index"].iloc[0]) if "Income Variability Index" in metrics_df.columns else 0.0
            salary_trend = float(metrics_df["Recent Salary Trend (%)"].iloc[0]) if "Recent Salary Trend (%)" in metrics_df.columns else 0.0
            savings_ratio = float(metrics_df["Savings Rate"].iloc[0]) if "Savings Rate" in metrics_df.columns else 0.0
            emi_payments = float(metrics_df["High-Cost EMI Payments"].iloc[0]) if "High-Cost EMI Payments" in metrics_df.columns else 0.0
            bank_score = results["heuristic_decision"].get("Total Score", 0)
            decision = results["heuristic_decision"]
            risk_level = decision.get("Risk Level", "Unknown")

            st.subheader("Analysis")
            st.write(f"- **Net Surplus**: ₹{monthly_income:,.2f} - ₹{monthly_expenses:,.2f} = ₹{net_surplus:,.2f}")
            st.write(f"- **Red Flags**: {red_flag_count} (including {bounced_cheques} bounced cheques)")
            st.write(f"- **Bank Statement Score**: {bank_score:.1f}/100")

            st.subheader("Decision")
            reason = results["final_reason"]
            st.write(f"**{results['final_action']}**: {reason}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Bank Score", f"{bank_score}/100")
            with col2:
                st.metric("Risk Level", risk_level)
            with col3:
                st.metric("Action", results['final_action'])

            st.subheader("ML Model Decision")
            ml_result = results["ml_result"]
            if ml_result:
                ml_prediction = ml_result["model_prediction"]
                ml_probability = ml_result["model_probability"]
                if ml_prediction == "APPROVED":
                    st.success(f"✅ {ml_prediction} ({ml_probability}% confidence)")
                else:
                    st.error(f"❌ {ml_prediction} ({ml_probability}% confidence)")
            else:
                st.warning("⚠️ ML Prediction Not Available")

            st.subheader("📈 Visual Insights")
            for plot_path, title in results["plots"]:
                if os.path.exists(plot_path):
                    st.image(plot_path, caption=title, use_column_width=True)

            pdf_path = results.get("pdf_path")
            if pdf_path and Path(pdf_path).exists():
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=f,
                        file_name=Path(pdf_path).name,
                        mime="application/pdf"
                    )
            else:
                st.warning("⚠️ PDF report not found.")

            output_file = OUTPUT_DIR / "loan_scores_log.csv"
            summary_data = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "File": file_name,
                "CIBIL Score": 720,
                "Bank Score": bank_score,
                "Risk Level": risk_level,
                "Action": results["final_action"],
                "Average Monthly Income": monthly_income,
                "Average Monthly Expenses": monthly_expenses,
                "Net Surplus": net_surplus,
                "DTI Ratio": dti_ratio,
                "Red Flag Count": red_flag_count,
                "Average Closing Balance": avg_monthly_balance,
                "Cash Withdrawals": cash_withdrawals,
                "Existing Loan Count": existing_loans,
                "Bounced Cheques Count": bounced_cheques,
                "Overdraft Usage Frequency": overdraft_frequency,
                "Negative Balance Days": negative_balance_days,
                "Sudden High-Value Credits": high_value_credits,
                "Circular Transactions": circular_transactions,
                "Income Variability Index": income_variability,
                "Recent Salary Trend (%)": salary_trend,
                "Savings Ratio (%)": savings_ratio,
                "High-Cost EMI Payments": emi_payments,
                "ML Decision": ml_result.get("model_prediction") if ml_result else "N/A",
                "Confidence": ml_result.get("model_probability") if ml_result else 0
            }
            df_output = pd.DataFrame([summary_data])
            if output_file.exists():
                df_output.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df_output.to_csv(output_file, index=False)

    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        logger.error(f"Analysis failed: {str(e)}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

    

    st.markdown('</div></div>', unsafe_allow_html=True)

