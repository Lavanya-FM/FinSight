import streamlit as st
from services.users import register_user, login_user
from services.files import save_uploaded_file, fetch_files
from services.reports import save_report, fetch_reports, update_report_decision
import tempfile
import os
import logging
import time
import json
import re
import io
import random
from datetime import datetime, timedelta, timezone
import extra_streamlit_components as stx
from typing import Optional
from services.base import client, admin_client
from supabase import create_client
import pandas as pd
import hashlib
import base64

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Setup logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="Loan Analytics System", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
def initialize_session_state():
    defaults = {
        "user_id": None,
        "file_id": None,
        "username": None,
        "role": None,
        "current_tab": None,
        "login_captcha": None,
        "register_captcha": None,
        "access_token": None,
        "user": None,
        "show_upload": False,
        "models_initialized": False,
        "models": {},
        "upload_counter": 0,
        "show_profile": False,
        "show_forgot_password": False,
        "show_reset_password": False, 
        "cibil_score": None,
        "show_cibil_input": False,
        "show_activity": False,
        "sidebar_expanded": False,
        "initial_load": True
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Call initialization at the start
initialize_session_state()

# Initialize other session state variables
if "access_token" not in st.session_state:
    st.session_state["access_token"] = None
if "user" not in st.session_state:
    st.session_state["user"] = None

# Supabase configuration
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE_KEY = st.secrets["SUPABASE_SERVICE_KEY"]
BUCKET_NAME = "files"
CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB per chunk
MAX_RETRIES = 3
RETRY_DELAY = 2

# Initialize Supabase clients
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
supabase_frontend = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

UPLOAD_STATE_FILE = "upload_state.json"

# Load CSS
def load_css(file_path):
    try:
        with open(file_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        logger.error(f"CSS file not found: {file_path}")
        st.error("Failed to load styles. Please ensure 'static/css/styles.css' exists.")

# Load JavaScript
def load_js(file_path, is_expanded):
    """Load JavaScript file with UTF-8 encoding."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            js_code = f.read()
            if is_expanded:
                js_code += "\ntoggleSidebar(true);"
            else:
                js_code += "\ntoggleSidebar(false);"
            return js_code
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode {file_path}: {str(e)}")
        return ""
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        return ""
    
# Load static assets
load_css("static/css/styles.css")
load_js("static/js/sidebar.js", st.session_state["sidebar_expanded"])
st.components.v1.html(f"<script>{load_js}</script>", height=0)

def load_upload_state():
    if os.path.exists(UPLOAD_STATE_FILE):
        with open(UPLOAD_STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_upload_state(state):
    with open(UPLOAD_STATE_FILE, "w") as f:
        json.dump(state, f)

# Helper functions
def calculate_file_hash(file_path: str) -> str:
    """Generate SHA256 hash for a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def save_file_metadata(user_id, filename, storage_path, file_size, file_type, file_hash):
    """
    Save metadata to uploaded_files table with error handling for duplicates.
    Returns the file_id if successful, None otherwise.
    """
    try:
        response = supabase.table("uploaded_files").insert({
            "file_name": filename,
            "file_path": storage_path,
            "file_size": file_size,
            "file_type": file_type,
            "user_id": user_id,
            "file_hash": file_hash,
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        
        if response.data and len(response.data) > 0:
            file_id = response.data[0].get('file_id')
            logging.info(f"✅ Metadata saved for {filename}: file_id={file_id}")
            return file_id
        else:
            logging.error(f"❌ No data returned from metadata insert for {filename}")
            return None
            
    except Exception as e:
        logging.error(f"❌ Metadata insert failed for {filename}: {str(e)}")
        return None

def save_report_with_text(user_id: str, file_id: int, decision: dict, report_text: str, plots_text: str) -> Optional[int]:
    """
    Saves report metadata to the database without storing files.
    Returns the report_id if successful, None otherwise.
    """
    try:
        current_time = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).strftime('%Y-%m-%d %H:%M:%S IST')
        logging.info(f"Starting report save process at {current_time}")

        # Prepare report details
        report_details = {
            "decision": decision,
            "generated_at": current_time,
            "report_text": report_text
        }

        # Extract decision components for separate columns
        action = decision.get("Action", "Unknown")
        reason = decision.get("Reason", "")
        confidence = decision.get("Confidence", None)

        # Use save_report from reports.py
        report_id = save_report(user_id, file_id, {
            "Action": action,
            "Reason": reason,
            "Confidence": confidence,
            "report_details": report_details
        })

        logging.info(f"✅ Report saved to database: report_id={report_id}")
        return report_id

    except Exception as e:
        logging.error(f"Failed to save report at {current_time}: {str(e)}")
        return None

def validate_user_id(user_id: str) -> bool:
    """Validate user_id format (assuming UUID format)"""
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, str(user_id).lower()))

def validate_action(action: str) -> bool:
    """Validate action is one of the expected values"""
    valid_actions = {'approve', 'approved', 'reject', 'rejected', 'pending', 'unknown'}
    return str(action).lower() in valid_actions

def test_database_connection():
    """Test basic database connectivity"""
    try:
        response = admin_client.table('users').select('user_id').limit(1).execute()
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

# Cookie manager for persistent login
cookie_manager = stx.CookieManager()
if not st.session_state["user_id"] and cookie_manager.get("user_id"):
    try:
        st.session_state["user_id"] = cookie_manager.get("user_id")
        user_data = client.table("users").select("user_id, username, role").eq("user_id", st.session_state["user_id"]).execute()
        if user_data.data:
            st.session_state.update({
                "username": user_data.data[0]["username"],
                "role": user_data.data[0]["role"],
                "current_tab": "home" if user_data.data[0]["role"] == "user" else "admin_reports"
            })
        else:
            cookie_manager.delete("user_id")
            st.session_state["user_id"] = None
    except Exception as e:
        logger.error(f"Failed to validate cookie: {str(e)}")
        cookie_manager.delete("user_id")
        st.session_state["user_id"] = None

# CAPTCHA generation
def generate_captcha():
    num1, num2 = random.randint(1, 10), random.randint(1, 10)
    operator = random.choice(['+', '-', '*'])
    answer = num1 + num2 if operator == '+' else num1 - num2 if operator == '-' else num1 * num2
    return {"question": f"{num1} {operator} {num2} = ?", "answer": str(answer).strip()}

# Initialize CAPTCHA if needed
if st.session_state["current_tab"] == "login" and not st.session_state["login_captcha"]:
    st.session_state["login_captcha"] = generate_captcha()
elif st.session_state["current_tab"] == "register" and not st.session_state["register_captcha"]:
    st.session_state["register_captcha"] = generate_captcha()

# Authenticate function
def authenticate(email: str, password: str) -> dict | None:
    try:
        user = login_user(email, password)
        if user:
            logger.info(f"User {email} authenticated successfully")
            return {"user_id": user[0], "name": user[1], "role": user[2]}
        logger.warning(f"Authentication failed for {email}")
        return None
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        return None

# Header
st.markdown("""
<div class="header">
    <div class="header-content">
        <h1>🏦 Loan Analytics System</h1>
        <p>Intelligent loan eligibility analysis with real-time insights</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar toggle logic
if st.button("☰", key="toggle_sidebar", help="Toggle Sidebar"):
    st.session_state["sidebar_expanded"] = not st.session_state["sidebar_expanded"]
    # Re-run to apply JavaScript with new state

# Check for recovery mode
query_params = st.query_params
if 'type' in query_params and query_params['type'] == 'recovery':
    st.session_state["show_reset_password"] = True
    st.session_state["current_tab"] = "reset_password"

# Landing page for non-logged-in users
if not st.session_state["user_id"]:
    st.markdown('<div class="content">', unsafe_allow_html=True)

    # Landing page
    st.markdown("""
    <div class="landing-section">
        <h1 class="landing-title">Transform Loan Decisions with AI</h1>
        <p class="landing-subtitle">Experience cutting-edge loan eligibility analysis powered by advanced AI. Upload documents, gain real-time insights, and make smarter lending decisions effortlessly.</p>
    </div>
    """, unsafe_allow_html=True)

    # Buttons inside the card
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Log In", key="landing_login", use_container_width=True):
            st.session_state["current_tab"] = "login"
            st.session_state["login_captcha"] = generate_captcha()
            st.rerun()
    
    with col2:
        if st.button("Get Started", key="landing_register", use_container_width=True):
            st.session_state["current_tab"] = "register"
            st.session_state["register_captcha"] = generate_captcha()
            st.rerun()

    # Handle password reset and forgot password flows
    if st.session_state["show_reset_password"]:
        st.markdown("### Reset Password")
        with st.form("reset_password_form"):
            new_password = st.text_input("New Password", type="password", placeholder="Enter new password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm new password")
            submitted = st.form_submit_button("Reset Password")
            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 8:
                    st.error("Password must be at least 8 characters")
                else:
                    try:
                        client.auth.update_user({'password': new_password})
                        st.success("Password reset successfully! Please log in.")
                        st.session_state["show_reset_password"] = False
                        st.session_state["current_tab"] = "login"
                        st.query_params.clear()
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to reset password: {str(e)}")

    elif st.session_state["show_forgot_password"]:
        st.markdown("### Forgot Password")
        with st.form("forgot_password_form"):
            email = st.text_input("Email Address", placeholder="Enter your email")
            submitted = st.form_submit_button("Send Reset Link")
            if submitted:
                if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    st.error("Invalid email format")
                else:
                    try:
                        client.auth.reset_password_for_email(email, options={'redirect_to': f"http://{st.runtime.get_instance()._get_host_address()}/?type=recovery"})
                        st.success("Password reset link sent to your email.")
                        st.session_state["show_forgot_password"] = False
                        st.session_state["current_tab"] = "login"
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to send reset link: {str(e)}")

    # Display forms based on button clicks
    else:
        if "current_tab" not in st.session_state:
            st.session_state["current_tab"] = None

        if st.session_state["current_tab"] == "login" or st.query_params.get("type") == "recovery":
            st.session_state["current_tab"] = "login"
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            st.markdown("### Welcome back!")
            with st.form("login_form", clear_on_submit=True):
                email = st.text_input("Email Address", placeholder="Enter your email")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                captcha = st.session_state.get("login_captcha", generate_captcha())
                st.session_state["login_captcha"] = captcha
                st.markdown(f'<div class="security-section"><div class="security-title">🔢 Security Verification:</div><div class="math-question">{captcha["question"]}</div></div>', unsafe_allow_html=True)
                security_answer = st.text_input("Answer", placeholder="Enter the answer").strip()
                submitted = st.form_submit_button("Sign In")

                if submitted:
                    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                        st.error("Invalid email format")
                    elif not password:
                        st.error("Password cannot be empty")
                    elif security_answer != captcha["answer"]:
                        st.error("Incorrect CAPTCHA answer")
                        st.session_state["login_captcha"] = generate_captcha()
                    else:
                        user = authenticate(email, password)
                        if user:
                            st.session_state.update({"user_id": user["user_id"], "username": user["name"], "role": user["role"], "is_premium": True})
                            try:
                                cookie_manager.set("user_id", user["user_id"], expires_at=datetime.now() + timedelta(days=7))
                            except Exception as e:
                                logger.error(f"Failed to set cookie: {str(e)}")
                            st.success("Login successful!")
                            st.session_state["login_captcha"] = None
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Invalid email or password")
                            st.session_state["login_captcha"] = generate_captcha()

            if st.button("Forgot Password?"):
                st.session_state["show_forgot_password"] = True
                st.session_state["current_tab"] = "forgot_password"
                st.rerun()

            if st.button("Create an Account"):
                st.session_state["current_tab"] = "register"
                st.session_state["register_captcha"] = generate_captcha()
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        elif st.session_state["current_tab"] == "register":
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            st.markdown("### Create Account")
            with st.form("register_form", clear_on_submit=True):
                username = st.text_input("Name", placeholder="Enter your name")
                email = st.text_input("Email Address", placeholder="Enter your email")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                st.markdown("**Choose your role:**")
                user_role = st.radio("Select Role", ["👤 User", "👑 Admin"], key="role_selection", label_visibility="collapsed")
                captcha = st.session_state.get("register_captcha", generate_captcha())
                st.session_state["register_captcha"] = captcha
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'<div class="security-section"><div class="security-title">🔢 Security Verification:</div><div class="math-question">{captcha["question"]}</div></div>', unsafe_allow_html=True)
                with col2:
                    security_answer = st.text_input("Answer", placeholder="Enter answer", key="security_answer")
                submitted = st.form_submit_button("Create Account")

                if submitted:
                    if not username:
                        st.error("Name cannot be empty")
                    elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                        st.error("Invalid email format")
                    elif len(password) < 8 or not any(c in "!@#$%^&*()_+" for c in password):
                        st.error("Password must be at least 8 characters and include a special character")
                    elif security_answer != captcha["answer"]:
                        st.error("Incorrect CAPTCHA answer")
                        st.session_state["register_captcha"] = generate_captcha()
                    else:
                        try:
                            role = "user" if user_role == "👤 User" else "admin"
                            register_user(username, email, password, role)
                            st.success("Account created successfully! Please log in.")
                            st.session_state["current_tab"] = "login"
                            st.session_state["register_captcha"] = None
                            st.session_state["login_captcha"] = generate_captcha()
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            if "duplicate key value violates unique constraint" in str(e):
                                st.error("Email already registered")
                            else:
                                st.error(f"Registration failed: {str(e)}")
                            st.session_state["register_captcha"] = generate_captcha()

            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'''
    <div class="footer">
        © 2025 Loan Analytics System. All rights reserved. | Last updated: {datetime.now().strftime("%I:%M %p IST on %B %d, %Y")} | System Status: Online ✅
    </div>
    ''', unsafe_allow_html=True)

else:
    # Header with profile section
    st.markdown(f"""
    <div class="header">
        <div class="header-content">
            <h1>🏦 Loan Analytics System</h1>
            <p>Intelligent loan eligibility analysis with real-time insights</p>
        </div>
        <div class="profile-section">
            <div class="profile-info">
                <div class="profile-name">{st.session_state["username"]}</div>
                <div class="profile-role">Last login: {datetime.now().strftime("%I:%M %p")}</div>
            </div>
            <div class="profile-badge {'admin' if st.session_state['role'] == 'admin' else ''}">{st.session_state["role"]}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-title">🏦 Loan Analytics</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        if st.session_state["role"] == "user":
            if st.button("🏠Home", key="home_btn", help="Go to Home"):
                st.session_state.update({"show_upload": False, "show_reports": False, "show_analytics": False, "show_profile": False, "show_activity": False})
            if st.button("👤Profile", key="profile_btn", help="View Profile"):
                st.session_state.update({"show_upload": False, "show_reports": False, "show_analytics": False, "show_profile": True, "show_activity": False})
            if st.button("📄Upload Documents", key="upload_btn", help="Upload Documents"):
                st.session_state["show_cibil_input"] = not st.session_state["show_cibil_input"]
                st.session_state.update({"show_upload": True, "show_reports": False, "show_analytics": False, "show_profile": False, "show_activity": False})

        if st.session_state["role"] == "admin":
            st.markdown("---")
            st.markdown("### Admin Tools")
            if st.button("📋 System Reports"):
                st.session_state.update({"current_tab": "admin_reports"})
                st.rerun()
            if st.button("👥 User Management"):
                st.session_state.update({"current_tab": "admin_users"})
                st.rerun()
            if st.button("⚙️ System Settings"):
                st.session_state.update({"current_tab": "admin_settings"})

        st.markdown("---")

        if st.button("🚪 Logout", key="sidebar_logout", use_container_width=True,help="Sign out"):
            try:
                client.auth.sign_out()
            except Exception as e:
                logger.error(f"Failed to sign out: {str(e)}")
            st.session_state.clear()
            initialize_session_state()  # Re-initialize session state
            try:
                cookie_manager.delete("user_id")
            except Exception as e:
                logger.error(f"Failed to delete cookie: {str(e)}")
            st.success("Logged out successfully")
 
            st.rerun() # Force rerun to reload the page in non-logged-in state
            
    # Main content area
    st.markdown('<div class="content">', unsafe_allow_html=True)

    if st.session_state["user_id"]:
        # Header with profile section for logged-in users
        st.markdown(f"""
        <div class="header">
            <div class="header-content">
                <h1>🏦 Loan Analytics System</h1>
                <p>Intelligent loan eligibility analysis with real-time insights</p>
            </div>
            <div class="profile-section">
                <div class="profile-info">
                    <div class="profile-name">{st.session_state["username"]}</div>
                    <div class="profile-role">Last login: {datetime.now().strftime("%I:%M %p")}</div>
                </div>
                <div class="profile-badge {'admin' if st.session_state['role'] == 'admin' else ''}">{st.session_state["role"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state["role"] == "user" and st.session_state["show_cibil_input"]:
            st.markdown('<div class="cibil-section">', unsafe_allow_html=True)
            st.markdown("### Enter CIBIL Score")
            if st.session_state.get("cibil_score") is not None:
                st.info(f"Current stored CIBIL score: {st.session_state['cibil_score']} (will be applied to next upload)")
            cibil_score = st.number_input("Your CIBIL Score", min_value=300, max_value=900, value=st.session_state.get("cibil_score", 300), key="center_cibil")
            if st.button("💾 Save CIBIL Score for Next Upload", key="save_center_cibil", use_container_width=True):
                try:
                    st.session_state["cibil_score"] = cibil_score
                    st.success("✅ CIBIL score saved temporarily. It will be associated with your next uploaded file.")
                except Exception as e:
                    logger.error(f"Failed to save CIBIL score to session state for user_id={st.session_state['user_id']}: {str(e)}")
                    st.error(f"❌ Failed to save CIBIL score: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'''
        <div class="welcome-section">
            <h2 class="welcome-title">Welcome, {st.session_state["username"]}</h2>
            <p class="welcome-subtitle">Manage your documents and unlock premium features via the sidebar</p>
        </div>
        ''', unsafe_allow_html=True)

        if st.session_state["show_profile"]:
            st.markdown("### Profile")
            with st.form("profile_form", clear_on_submit=True):
                new_username = st.text_input("Name", value=st.session_state["username"])
                new_email = st.text_input("Email", value=client.table("users").select("email").eq("user_id", st.session_state["user_id"]).execute().data[0]["email"])
                new_password = st.text_input("New Password (leave blank to keep current)", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                submitted = st.form_submit_button("Update Profile")
                if submitted:
                    update_dict = {}
                    if new_username != st.session_state["username"]:
                        update_dict["username"] = new_username
                    if new_password:
                        if new_password != confirm_password:
                            st.error("Passwords do not match")
                        else:
                            client.auth.update_user({'password': new_password})
                            st.success("Password updated")

        elif st.session_state["role"] == "user":
            st.markdown('<div class="dashboard-section"><h3 class="section-title">Document Analysis Center</h3></div>', unsafe_allow_html=True)
        # Include JavaScript
            js_code = load_js("static/js/sidebar.js", st.session_state["sidebar_expanded"])
            st.components.v1.html(f"<script>{js_code}</script>", height=0)

            if st.session_state["show_upload"]:
                uploaded_files = st.file_uploader(
                    "Choose Files",
                    type=['pdf', 'csv', 'xlsx', 'txt', 'doc', 'docx'],
                    help="Drag and drop files here • Limit 200MB per file • PDF, CSV, XLSX, TXT, DOC, DOCX",
                    accept_multiple_files=True
                )
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        try:
                            file_id = save_uploaded_file(st.session_state["user_id"], uploaded_file.name, tmp_path)
                            files = fetch_files(st.session_state["user_id"])

                            if any(f["file_id"] == file_id for f in files):
                                st.success(f"File uploaded: {uploaded_file.name}")
                                if st.session_state.get("cibil_score") is not None:
                                    try:
                                        client.table("uploaded_files").update({"cibil_score": st.session_state["cibil_score"]}).eq("file_id", file_id).execute()
                                        st.success(f"✅ CIBIL score {st.session_state['cibil_score']} associated with file: {uploaded_file.name}")
                                        st.session_state["file_id"] = file_id
                                    except Exception as e:
                                        logger.error(f"Failed to associate CIBIL score with file_id={file_id}: {str(e)}")
                                        st.error(f"❌ Failed to associate CIBIL score: {str(e)}")
                                try:
                                    from visualizer import analyze_file
                                    with st.spinner("Analyzing file..."):
                                        analysis_result = analyze_file(tmp_path)
                                        if isinstance(analysis_result, dict):
                                            decision_json_path = analysis_result.get("decision_json")
                                            report_text = analysis_result.get("report_text", "No report text generated")
                                            plots_text = analysis_result.get("plots_text", "No plots text generated")

                                            if decision_json_path and os.path.exists(decision_json_path):
                                                try:
                                                    with open(decision_json_path, "r", encoding="utf-8") as f:
                                                        decision_data = json.load(f)
                                                    action = decision_data.get("heuristic_action", "Unknown").capitalize()
                                                    reason = decision_data.get("heuristic_reason", "No reason provided")
                                                    confidence = decision_data.get("confidence", None)
                                                    decision = {
                                                        "Action": action,
                                                        "Reason": reason,
                                                        "Confidence": confidence
                                                    }
                                                except Exception as e:
                                                    logger.error(f"Failed to parse decision JSON: {str(e)}")
                                                    decision = {"Action": "Unknown", "Reason": f"Failed to parse decision JSON: {str(e)}", "Confidence": None}
                                            else:
                                                decision = {"Action": "Unknown", "Reason": "No decision JSON found", "Confidence": None}

                                            report_id = save_report_with_text(
                                                st.session_state["user_id"],
                                                file_id,
                                                decision,
                                                report_text,
                                                plots_text
                                            )

                                            if report_id:
                                                st.session_state["upload_counter"] += 1
                                                st.success(f"✅ Analysis complete! Report saved with ID: {report_id}")
                                            else:
                                                st.error("Failed to save analysis results")
                                except ImportError:
                                    st.error("Analysis module not found. Please ensure 'visualizer.py' with 'analyze_file' function exists.")
                                except Exception as e:
                                    st.error(f"Analysis failed: {str(e)}")
                                    logger.error(f"Analysis error for user {st.session_state['user_id']}: {str(e)}")
                            else:
                                st.error("File upload failed")
                        except Exception as e:
                            st.error(f"Upload error: {str(e)}")
                            logger.error(f"Upload error for user {st.session_state['user_id']}: {str(e)}")
                        finally:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)

    elif st.session_state["role"] == "admin" and st.session_state["user_id"]:     
        # Admin Dashboard (only shown if logged in as admin)   
        st.markdown('<div class="dashboard-section"><h3 class="section-title">Admin Control Panel</h3></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 System Reports", "👥 User Management", "⚙️ Settings"])

        with tab1:
            st.markdown("### 📈 All System Reports")
            try:
                reports = fetch_reports()
                if reports:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Reports", len(reports))
                    with col2:
                        approved_count = sum(1 for r in reports if isinstance(r.get("decision"), dict) and r["decision"].get("Action") == "Approve")
                        st.metric("Approved", approved_count)
                    with col3:
                        rejected_count = sum(1 for r in reports if isinstance(r.get("decision"), dict) and r["decision"].get("Action") == "Reject")
                        st.metric("Rejected", rejected_count)
                    with col4:
                        review_count = len(reports) - approved_count - rejected_count
                        st.metric("Under Review", review_count)
                    
                    st.markdown("### Recent Reports")
                    for report in reports[-10:]:
                        decision_dict = report.get("decision", {"Action": "Unknown"})
                        action = decision_dict.get("Action", "Unknown") if isinstance(decision_dict, dict) else "Unknown"
                        reason = decision_dict.get("Reason", "") if isinstance(decision_dict, dict) else ""
                        with st.expander(f"Report #{report['report_id']} - User {report['user_id']}", expanded=False):
                            col_left, col_right = st.columns(2)
                            with col_left:
                                st.write(f"**Decision:** {action}")
                                st.write(f"**Reason:** {reason}")
                                st.write(f"**User ID:** {report['user_id']}")
                                st.write(f"**Generated:** {report['generated_at']}")
                            with col_right:
                                pass
                else:
                    st.info("No reports found in the system.")
            except Exception as e:
                st.error(f"❌ Failed to fetch reports: {str(e)}")

            st.markdown("### 🔍 Manual Review & Override")
            with st.form("review_form"):
                col1, col2 = st.columns(2)
                with col1:
                    report_id = st.number_input("Report ID to Review", min_value=1, step=1, value=1)
                    decision = st.selectbox("Set Decision", ["Approve", "Reject", "Review Manually"])
                with col2:
                    reason = st.text_area("Reason for Override", placeholder="Enter reason for manual override...")
                
                submitted = st.form_submit_button("✅ Update Report Decision", use_container_width=True)
                
                if submitted:
                    try:
                        update_report_decision(report_id, {"Action": decision, "Reason": reason}, st.session_state["user_id"], reason)
                        st.success(f"✅ Report {report_id} updated to: {decision}")
                    except Exception as e:
                        st.error(f"❌ Failed to update report: {str(e)}")

        with tab2:
            st.markdown("### 👥 User Management")
            try:
                users = client.table("users").select("user_id, username, email, role, created_at").execute()
                if users.data:
                    total_users = len(users.data)
                    admin_count = sum(1 for u in users.data if u['role'] == 'admin')
                    user_count = total_users - admin_count
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Users", total_users)
                    with col2:
                        st.metric("Regular Users", user_count)
                    with col3:
                        st.metric("Administrators", admin_count)
                    for i, user in enumerate(users.data):
                        with st.expander(f"👤 {user['username']} ({user['role'].title()})", expanded=False):
                            col_left, col_right = st.columns(2)
                            with col_left:
                                st.write(f"**User ID:** {user['user_id']}")
                                st.write(f"**Email:** {user['email']}")
                            with col_right:
                                st.write(f"**Role:** {user['role'].title()}")
                                if 'created_at' in user:
                                    st.write(f"**Joined:** {user['created_at']}")
                else:
                    st.info("No users found in the system.")
            except Exception as e:
                st.error(f"Failed to fetch users: {str(e)}")
            st.markdown("### ➕ Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Export User List", use_container_width=True):
                    st.info("User export functionality would be implemented here.")
            with col2:
                if st.button("📧 Send System Notification", use_container_width=True):
                    st.info("Notification system would be implemented here.")

        with tab3:
            st.markdown("### ⚙️ System Configuration")
            with st.form("settings_form"):
                st.markdown("#### File Upload Settings")
                col1, col2 = st.columns(2)
                with col1:
                    max_file_size_mb = st.number_input("Max file size (MB)", min_value=1, max_value=500, value=200)
                    analysis_timeout = st.number_input("Analysis timeout (minutes)", min_value=1, max_value=60, value=10)
                with col2:
                    auto_approve_threshold = st.slider("Auto-approve confidence threshold", 0.0, 1.0, 0.85, 0.05)
                    enable_notifications = st.checkbox("Enable email notifications", value=True)
                st.markdown("#### Security Settings")
                col1, col2 = st.columns(2)
                with col1:
                    session_timeout = st.number_input("Session timeout (hours)", min_value=1, max_value=24, value=8)
                    require_captcha = st.checkbox("Require CAPTCHA for login", value=True)
                with col2:
                    max_login_attempts = st.number_input("Max login attempts", min_value=3, max_value=10, value=5)
                    enable_audit_log = st.checkbox("Enable audit logging", value=True)
                st.markdown("#### AI Model Settings")
                col1, col2 = st.columns(2)
                with col1:
                    model_version = st.selectbox("AI Model Version", ["v2.1", "v2.0", "v1.9"], index=0)
                    enable_advanced_analysis = st.checkbox("Enable advanced risk analysis", value=True)
                with col2:
                    batch_processing = st.checkbox("Enable batch processing", value=False)
                    quality_threshold = st.slider("Document quality threshold", 0.0, 1.0, 0.75, 0.05)
                submitted = st.form_submit_button("💾 Save Configuration", use_container_width=True)
                if submitted:
                    st.success("✅ System settings saved successfully!")
                    st.info("⚠️ Some changes may require system restart to take effect.")
    else:
        # Landing page for non-logged-in users
        st.markdown('<div class="content">', unsafe_allow_html=True)

        # Landing page
        st.markdown("""
        <div class="landing-section">
            <h1 class="landing-title">Transform Loan Decisions with AI</h1>
            <p class="landing-subtitle">Experience cutting-edge loan eligibility analysis powered by advanced AI. Upload documents, gain real-time insights, and make smarter lending decisions effortlessly.</p>
        </div>
        """, unsafe_allow_html=True)

        # Buttons inside the card
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Log In", key="landing_login", use_container_width=True):
                st.session_state["current_tab"] = "login"
                st.session_state["login_captcha"] = generate_captcha()
                st.rerun()

        with col2:
            if st.button("Get Started", key="landing_register", use_container_width=True):
                st.session_state["current_tab"] = "register"
                st.session_state["register_captcha"] = generate_captcha()
                st.rerun()

        # Handle password reset and forgot password flows
        if st.session_state["show_reset_password"]:
            st.markdown("### Reset Password")
            with st.form("reset_password_form"):
                new_password = st.text_input("New Password", type="password", placeholder="Enter new password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm new password")
                submitted = st.form_submit_button("Reset Password")
                if submitted:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 8:
                        st.error("Password must be at least 8 characters")
                    else:
                        try:
                            client.auth.update_user({'password': new_password})
                            st.success("Password reset successfully! Please log in.")
                            st.session_state["show_reset_password"] = False
                            st.session_state["current_tab"] = "login"
                            st.query_params.clear()
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to reset password: {str(e)}")

        elif st.session_state["show_forgot_password"]:
            st.markdown("### Forgot Password")
            with st.form("forgot_password_form"):
                email = st.text_input("Email Address", placeholder="Enter your email")
                submitted = st.form_submit_button("Send Reset Link")
                if submitted:
                    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                        st.error("Invalid email format")
                    else:
                        try:
                            client.auth.reset_password_for_email(email, options={'redirect_to': f"http://{st.runtime.get_instance()._get_host_address()}/?type=recovery"})
                            st.success("Password reset link sent to your email.")
                            st.session_state["show_forgot_password"] = False
                            st.session_state["current_tab"] = "login"
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to send reset link: {str(e)}")

        # Display forms based on button clicks
        else:
            if "current_tab" not in st.session_state:
                st.session_state["current_tab"] = None

            if st.session_state["current_tab"] == "login" or st.query_params.get("type") == "recovery":
                st.session_state["current_tab"] = "login"
                st.markdown('<div class="auth-container">', unsafe_allow_html=True)
                st.markdown("### Welcome back!")
                with st.form("login_form", clear_on_submit=True):
                    email = st.text_input("Email Address", placeholder="Enter your email")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")
                    captcha = st.session_state.get("login_captcha", generate_captcha())
                    st.session_state["login_captcha"] = captcha
                    st.markdown(f'<div class="security-section"><div class="security-title">🔢 Security Verification:</div><div class="math-question">{captcha["question"]}</div></div>', unsafe_allow_html=True)
                    security_answer = st.text_input("Answer", placeholder="Enter the answer").strip()
                    submitted = st.form_submit_button("Sign In")

                    if submitted:
                        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                            st.error("Invalid email format")
                        elif not password:
                            st.error("Password cannot be empty")
                        elif security_answer != captcha["answer"]:
                            st.error("Incorrect CAPTCHA answer")
                            st.session_state["login_captcha"] = generate_captcha()
                        else:
                            user = authenticate(email, password)
                            if user:
                                st.session_state.update({"user_id": user["user_id"], "username": user["name"], "role": user["role"], "is_premium": True})
                                try:
                                    cookie_manager.set("user_id", user["user_id"], expires_at=datetime.now() + timedelta(days=7))
                                except Exception as e:
                                    logger.error(f"Failed to set cookie: {str(e)}")
                                st.success("Login successful!")
                                st.session_state["login_captcha"] = None
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Invalid email or password")
                                st.session_state["login_captcha"] = generate_captcha()

                if st.button("Forgot Password?"):
                    st.session_state["show_forgot_password"] = True
                    st.session_state["current_tab"] = "forgot_password"
                    st.rerun()

                if st.button("Create an Account"):
                    st.session_state["current_tab"] = "register"
                    st.session_state["register_captcha"] = generate_captcha()
                    st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

            elif st.session_state["current_tab"] == "register":
                st.markdown('<div class="auth-container">', unsafe_allow_html=True)
                st.markdown("### Create Account")
                with st.form("register_form", clear_on_submit=True):
                    username = st.text_input("Name", placeholder="Enter your name")
                    email = st.text_input("Email Address", placeholder="Enter your email")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")
                    st.markdown("**Choose your role:**")
                    user_role = st.radio("Select Role", ["👤 User", "👑 Admin"], key="role_selection", label_visibility="collapsed")
                    captcha = st.session_state.get("register_captcha", generate_captcha())
                    st.session_state["register_captcha"] = captcha
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f'<div class="security-section"><div class="security-title">🔢 Security Verification:</div><div class="math-question">{captcha["question"]}</div></div>', unsafe_allow_html=True)
                    with col2:
                        security_answer = st.text_input("Answer", placeholder="Enter answer", key="security_answer")
                    submitted = st.form_submit_button("Create Account")

                    if submitted:
                        if not username:
                            st.error("Name cannot be empty")
                        elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                            st.error("Invalid email format")
                        elif len(password) < 8 or not any(c in "!@#$%^&*()_+" for c in password):
                            st.error("Password must be at least 8 characters and include a special character")
                        elif security_answer != captcha["answer"]:
                            st.error("Incorrect CAPTCHA answer")
                            st.session_state["register_captcha"] = generate_captcha()
                        else:
                            try:
                                role = "user" if user_role == "👤 User" else "admin"
                                register_user(username, email, password, role)
                                st.success("Account created successfully! Please log in.")
                                st.session_state["current_tab"] = "login"
                                st.session_state["register_captcha"] = None
                                st.session_state["login_captcha"] = generate_captcha()
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                if "duplicate key value violates unique constraint" in str(e):
                                    st.error("Email already registered")
                                else:
                                    st.error(f"Registration failed: {str(e)}")
                                st.session_state["register_captcha"] = generate_captcha()

                if st.button("Back to Login"):
                    st.session_state["current_tab"] = "login"
                    st.session_state["login_captcha"] = generate_captcha()
                    st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

#Footer
st.markdown(f'''
<div class="footer">
    © 2025 Loan Analytics System. All rights reserved.
    | Last updated: {datetime.now().strftime("%I:%M %p IST on %B %d, %Y")} | System Status: Online ✅
</div>
''', unsafe_allow_html=True)