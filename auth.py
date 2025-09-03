import streamlit as st
from app import display_user_dashboard  # Import the user dashboard function from app.py
from services.files import save_uploaded_file, fetch_files
from services.reports import save_report, fetch_reports, update_report_decision
import tempfile
import os
import logging
import json
import time
from datetime import datetime
from services.base import client ,ensure_bucket , ensure_session# Import client from services.base
from visualizer import analyze_file 

# Setup logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Authentication and routing flow
def auth_flow(user_id, role):
    if client is None:
        st.error("Database connection failed. Please contact support.")
        return

    try:
        ensure_bucket()  # Ensure the 'files' bucket exists
    except Exception as e:
        st.error(f"Failed to ensure storage bucket: {str(e)}. Please check Supabase configuration.")
        logger.error(f"Bucket creation failed: {str(e)}")
        return
    
    if role == "user":
        # User Dashboard
        display_user_dashboard(user_id)
    
        if st.session_state.get("show_upload", False):
            st.subheader("Upload Bank Statement")
            user_id = st.session_state.get("user_id")
            if not user_id:
                st.error("Please log in to upload files.")
                logger.error("No user_id in session for upload")
                return
            
            # Use a unique key to avoid multiple uploader instances
            uploaded_file = st.file_uploader("Choose file", type=['pdf', 'csv', 'xlsx', 'txt', 'doc', 'docx'], key="upload_file_" + str(time.time()), label_visibility="hidden")     
            if uploaded_file:
                if not uploaded_file.name:
                    st.error("File name cannot be empty.")
                    logger.error("Empty file name provided")
                    return
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                    logger.info(f"Created temporary file: {tmp_path}")
                try:
                    logger.info(f"Uploading file: {uploaded_file.name} for user: {user_id}")
                    file_id = save_uploaded_file(user_id, uploaded_file.name, tmp_path)
                    files = fetch_files(user_id)
                    if any(f["file_id"] == file_id for f in files):
                        st.success(f"📂 File uploaded successfully (ID: {file_id})")
                        if st.button("🚀 Run Analysis"):
                            try:
                                file_path = next(f["file_path"] for f in files if f["file_id"] == file_id)
                                response = client.storage.from_('files').download(file_path)
                                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                                    tmp.write(response)
                                    analysis_tmp_path = tmp.name
                                    logger.info(f"Created analysis temp file: {analysis_tmp_path}")
                                with st.spinner("Analyzing..."):
                                    results = analyze_file(analysis_tmp_path, cibil_score=720, out_dir="outputs")

                                    decision_json = results.get("decision_json")
                                    report_path = results.get("pdf_report")

                                    decision = {"Action": "Reject", "Reason": "Analysis failed"}
                                    if decision_json and os.path.exists(decision_json):
                                        with open(decision_json, "r", encoding="utf-8") as f:
                                            decision = json.load(f)

                                if report_path and os.path.exists(report_path):
                                    with open(report_path, 'rb') as f:
                                        report_storage_path = f"user_{user_id}/reports/{os.path.basename(report_path)}"
                                        logger.info(f"Uploading report to: {report_storage_path}")
                                        client.storage.from_('files').upload(
                                            report_storage_path,
                                            f.read(),
                                            {'content-type': 'application/pdf'}
                                        )
                                    report_id = save_report(user_id, file_id, decision, report_storage_path)
                                    st.success(f"✅ Analysis complete (Report ID: {report_id}, Status: {decision.get('Action')})")
                                else:
                                    st.error("❌ No report generated")

                                if os.path.exists(analysis_tmp_path):
                                    os.unlink(analysis_tmp_path)
                                    logger.info(f"Deleted analysis temp file: {analysis_tmp_path}")
                            except Exception as e:
                                st.error(f"❌ Analysis failed: {str(e)}")
                                logger.error(f"Analysis failed: {str(e)}")
                    else:
                        st.error("❌ File upload failed")
                except Exception as e:
                    st.error(f"❌ Upload error: {str(e)}")
                    logger.error(f"Upload error: {str(e)}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        logger.info(f"Deleted temporary file: {tmp_path}")
                    else:
                        logger.warning(f"Temporary file not found for deletion: {tmp_path}")

        
    elif role == "admin":
        # Admin Dashboard
        st.markdown('<div style="margin: 20px;"><div style="background: rgba(255,255,255,0.1); backdrop-filter: blur(20px); border-radius: 16px; padding: 24px; border: 1px solid rgba(255,255,255,0.2);"><h2 style="color: white; margin: 0 0 16px 0;">👑 Admin Dashboard</h2><div class="dashboard-grid"><div class="dashboard-card"><div class="card-icon">👥</div><div class="card-title">User Management</div><div class="card-description">Manage user accounts and permissions</div></div><div class="dashboard-card"><div class="card-icon">📊</div><div class="card-title">All Reports</div><div class="card-description">View comprehensive system reports</div></div><div class="dashboard-card"><div class="card-icon">🔧</div><div class="card-title">Manual Review</div><div class="card-description">Review and approve pending cases</div></div><div class="dashboard-card"><div class="card-icon">⚙️</div><div class="card-title">System Settings</div><div class="card-description">Configure system parameters</div></div></div></div></div>', unsafe_allow_html=True)

        # Admin functionality tabs
        tab1, tab2, tab3 = st.tabs(["📊 Reports", "👥 Users", "⚙️ Settings"])

        with tab1:
            st.markdown('<h3 style="color: #ffffff;">All Reports</h3>', unsafe_allow_html=True)
            try:
                reports = fetch_reports()
                if reports:
                    for report in reports:
                        st.markdown(f'<p style="color: #ffffff;">ID: {report["report_id"]}, User: {report["user_id"]}, Decision: {report["decision"]}, Date: {report["generated_at"]}</p>', unsafe_allow_html=True)
                        if st.button(f"Download Report {report['report_id']}", key=f"admin_download_{report['report_id']}"):
                            response = client.storage.from_(report["report_path"]).download()
                            st.download_button(label="📥 Download", data=response, file_name=f"report_{report['report_id']}.pdf", mime="application/pdf")
                else:
                    st.info("No reports found.")
            except Exception as e:
                st.error(f"❌ Failed to fetch reports: {str(e)}")

            st.markdown('<h3 style="color: #ffffff;">🔍 Manual Review</h3>', unsafe_allow_html=True)
            report_id = st.number_input("Enter Report ID to Review", min_value=1, step=1)
            decision = st.selectbox("Set Decision", ["Approved", "Rejected", "Review Manually"])
            reason = st.text_area("Reason for override")
            if st.button("🔧 Update Report Decision"):
                try:
                    update_report_decision(report_id, decision, user_id, reason)
                    st.success(f"✅ Report {report_id} updated to: {decision}")
                except Exception as e:
                    st.error(f"❌ Failed to update report: {str(e)}")

        with tab2:
            st.markdown('<h3 style="color: #ffffff;">User Management</h3>', unsafe_allow_html=True)
            try:
                users = client.table("users").select("user_id, username, email, role").execute()
                if users.data:
                    for user in users.data:
                        st.markdown(f'<p style="color: #ffffff;">ID: {user["user_id"]}, Name: {user["username"]}, Email: {user["email"]}, Role: {user["role"]}</p>', unsafe_allow_html=True)
                else:
                    st.info("No users found.")
            except Exception as e:
                st.error(f"❌ Failed to fetch users: {str(e)}")

        with tab3:
            st.markdown('<h3 style="color: #ffffff;">System Settings</h3>', unsafe_allow_html=True)
            max_file_size_mb = st.number_input("Max file size (MB)", min_value=1, max_value=100, value=10)
            if st.button("Save Settings"):
                st.success(f"Settings saved: Max file size set to {max_file_size_mb} MB")