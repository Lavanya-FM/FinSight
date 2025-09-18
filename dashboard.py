import os
import logging
import subprocess
import signal
import time
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from dotenv import load_dotenv
from typing import Optional
import requests
from urllib.parse import urlencode
from pathlib import Path

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if SUPABASE_URL and SUPABASE_ANON_KEY:
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    admin_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_SERVICE_KEY else None
else:
    client = None
    admin_client = None
    logger.warning("Supabase not configured properly")

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Streamlit configuration
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"
streamlit_process = None

def start_streamlit():
    global streamlit_process
    if streamlit_process is None or streamlit_process.poll() is not None:
        try:
            streamlit_process = subprocess.Popen([
                "streamlit", "run", "dashboard.py",
                "--server.port", str(STREAMLIT_PORT),
                "--server.address", "0.0.0.0",  # Allow external access
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"Started Streamlit on port {STREAMLIT_PORT}")
            time.sleep(5)  # Give it time to start
        except Exception as e:
            logger.error(f"Failed to start Streamlit: {str(e)}")

def stop_streamlit():
    global streamlit_process
    if streamlit_process:
        try:
            streamlit_process.terminate()
            streamlit_process.wait(timeout=5)
            logger.info("Streamlit server stopped")
        except subprocess.TimeoutExpired:
            streamlit_process.kill()
            logger.info("Streamlit server killed")
        except Exception as e:
            logger.error(f"Error stopping Streamlit: {str(e)}")

def start_streamlit_with_retry():
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            start_streamlit()
            time.sleep(3)
            response = requests.get(f"{STREAMLIT_URL}/_stcore/health", timeout=5)
            if response.status_code == 200:
                logger.info("Streamlit startup successful")
                break
            else:
                logger.warning(f"Streamlit health check failed on attempt {attempt + 1}")
        except Exception as e:
            logger.error(f"Streamlit startup attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_attempts - 1:
                time.sleep(2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("FastAPI app starting up...")
    threading.Thread(target=start_streamlit_with_retry, daemon=True).start()
    yield
    # Shutdown
    logger.info("FastAPI app shutting down...")
    stop_streamlit()

# Create FastAPI app with explicit prefix handling
app = FastAPI(
    title="FinSight API",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Project paths
PROJECT_ROOT = Path(__file__).parent
BUILD_DIR = PROJECT_ROOT / "finsight-ui" / "build"
INDEX_HTML_PATH = BUILD_DIR / "index.html"

def get_user_session_with_token(token: str = None):
    try:
        if not client or not admin_client:
            return None
            
        if token:
            temp_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            temp_client.auth.set_session(token, token)
            session_response = temp_client.auth.get_session()
        else:
            session_response = client.auth.get_session()
            
        if session_response and hasattr(session_response, 'data') and session_response.data and session_response.data.session:
            user_id = session_response.data.session.user.id
            user_data = admin_client.table("users").select("user_id, username, role").eq("user_id", user_id).execute()
            if user_data.data and len(user_data.data) > 0:
                logger.info(f"Session valid for user: {user_data.data[0]['username']}")
                return {
                    "user_id": user_data.data[0]["user_id"],
                    "username": user_data.data[0]["username"],
                    "role": user_data.data[0]["role"],
                    "session": session_response.data.session
                }
        return None
    except Exception as e:
        logger.error(f"Session check failed: {str(e)}")
        return None

def get_user_session_from_request(request: Request):
    try:
        token = request.cookies.get("supabase_token")
        if token:
            logger.info("Found token in cookie")
            return get_user_session_with_token(token)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            logger.info("Found token in Authorization header")
            return get_user_session_with_token(token)
        logger.info("No authentication token found")
        return None
    except Exception as e:
        logger.error(f"Session check from request failed: {str(e)}")
        return None

def is_streamlit_ready():
    try:
        response = requests.get(f"{STREAMLIT_URL}/_stcore/health", timeout=3)
        return response.status_code == 200
    except Exception:
        return False

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} for {request.method} {request.url.path} in {process_time:.2f}s")
    return response

# Root path - simple test
@app.get("/")
async def root():
    return {"message": "FinSight API is running", "status": "ok"}

# API Routes with explicit paths
@app.get("/api/check-session")
async def check_session_api(request: Request):
    logger.info("API check-session endpoint called")
    try:
        user_session = get_user_session_from_request(request)
        if user_session:
            logger.info(f"Session check successful for user: {user_session['username']}")
            return JSONResponse(content={
                "authenticated": True,
                "user": {
                    "userId": user_session["user_id"],
                    "username": user_session["username"],
                    "role": user_session["role"]
                }
            })
        else:
            logger.info("No valid session found in check-session")
            return JSONResponse(content={
                "authenticated": False,
                "user": None
            }, status_code=401)
    except Exception as e:
        logger.error(f"Session check API error: {str(e)}")
        return JSONResponse(content={
            "authenticated": False,
            "user": None,
            "error": str(e)
        }, status_code=500)

@app.post("/api/logout")
async def logout_api(request: Request):
    logger.info("API logout endpoint called")
    try:
        user_session = get_user_session_from_request(request)
        if user_session and client:
            token = request.cookies.get("supabase_token")
            if token:
                temp_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
                temp_client.auth.set_session(token, token)
                temp_client.auth.sign_out()
        logger.info("User logged out successfully")
        response = JSONResponse(content={"status": "success"})
        response.delete_cookie("supabase_token")
        response.delete_cookie("streamlit_session_token")
        return response
    except Exception as e:
        logger.error(f"Failed to sign out from Supabase: {str(e)}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/streamlit-health")
async def streamlit_health_api():
    logger.info("API streamlit-health endpoint called")
    return {"ready": is_streamlit_ready()}

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    try:
        streamlit_ready = is_streamlit_ready()
        return {
            "status": "healthy",
            "streamlit": "running" if streamlit_ready else "not running",
            "streamlit_url": STREAMLIT_URL,
            "supabase_configured": bool(SUPABASE_URL and SUPABASE_ANON_KEY),
            "build_exists": BUILD_DIR.exists(),
            "base_url": BASE_URL
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/debug/session")
async def debug_session(request: Request):
    logger.info("Debug session endpoint called")
    user_session = get_user_session_from_request(request)
    return {
        "session_found": bool(user_session),
        "user_session": user_session,
        "cookies": dict(request.cookies),
        "headers": dict(request.headers),
        "streamlit_url": STREAMLIT_URL,
        "streamlit_ready": is_streamlit_ready(),
        "base_url": BASE_URL
    }

# Dashboard route
@app.get("/dashboard")
async def serve_dashboard(request: Request):
    logger.info("Dashboard endpoint called")
    logger.info(f"Request cookies: {list(request.cookies.keys())}")
    
    user_session = get_user_session_from_request(request)
    if not user_session:
        logger.warning("No valid session for dashboard, redirecting to login")
        return RedirectResponse(url="/login", status_code=302)
    
    logger.info(f"Valid session for user {user_session['username']}, checking Streamlit status")
    
    if not is_streamlit_ready():
        logger.warning("Streamlit not ready, attempting to start...")
        start_streamlit()
        for i in range(10):
            time.sleep(1)
            if is_streamlit_ready():
                logger.info("Streamlit is now ready")
                break
        else:
            logger.error("Streamlit failed to start within timeout")
            return HTMLResponse("""
            <html><body><div style="text-align: center; padding: 2rem; font-family: Arial, sans-serif;">
                <h2 style="color: #e74c3c;">Dashboard Service Unavailable</h2>
                <p>The dashboard service is temporarily unavailable. Please try again in a few moments.</p>
                <button onclick="window.location.reload()">Retry</button>
                <script>setTimeout(function() { window.location.reload(); }, 5000);</script>
            </div></body></html>
            """, status_code=503)
    
    params = {
        "user_id": user_session["user_id"],
        "username": user_session["username"],
        "role": user_session["role"],
        "ref": "fastapi"
    }
    streamlit_url_with_params = f"{STREAMLIT_URL}?{urlencode(params)}"
    logger.info(f"Redirecting to Streamlit: {streamlit_url_with_params}")
    
    redirect_response = RedirectResponse(url=streamlit_url_with_params, status_code=302)
    redirect_response.set_cookie("streamlit_user_id", user_session["user_id"], httponly=False, secure=False, samesite="lax")
    redirect_response.set_cookie("streamlit_username", user_session["username"], httponly=False, secure=False, samesite="lax")
    redirect_response.set_cookie("streamlit_role", user_session["role"], httponly=False, secure=False, samesite="lax")
    return redirect_response

# Serve Streamlit dashboard within iframe
@app.get("/dashboard/streamlit")
async def serve_streamlit_dashboard(request: Request):
    logger.info("Streamlit dashboard endpoint called")
    user_session = get_user_session_from_request(request)
    if not user_session:
        logger.warning("No valid session, redirecting to login")
        return RedirectResponse(url="/login", status_code=302)

    if not is_streamlit_ready():
        logger.warning("Streamlit not ready, attempting to start...")
        start_streamlit()
        for i in range(10):
            time.sleep(1)
            if is_streamlit_ready():
                logger.info("Streamlit is now ready")
                break
        else:
            logger.error("Streamlit failed to start within timeout")
            return HTMLResponse("""
            <html><body><div style="text-align: center; padding: 2rem;">
                <h2>Dashboard Unavailable</h2>
                <p>Please try again in a moment.</p>
                <button onclick="location.reload()">Retry</button>
            </div></body></html>
            """, status_code=503)

    params = {
        "user_id": user_session["user_id"],
        "username": user_session["username"],
        "role": user_session["role"]
    }
    streamlit_url = f"{STREAMLIT_URL}?{urlencode(params)}"
    return HTMLResponse(f"""
    <html>
      <body style="margin: 0; padding: 0; overflow: hidden;">
        <iframe src="{streamlit_url}" style="width: 100%; height: 100vh; border: none;" title="Streamlit Dashboard"></iframe>
      </body>
    </html>
    """)

# Static file handling
if BUILD_DIR.exists():
    # Serve favicon
    @app.get("/favicon.ico")
    async def favicon():
        from fastapi.responses import FileResponse
        favicon_path = BUILD_DIR / "favicon.ico"
        if favicon_path.exists():
            return FileResponse(favicon_path)
        raise HTTPException(status_code=404)
    
    # Serve manifest
    @app.get("/manifest.json")
    async def manifest():
        from fastapi.responses import FileResponse
        manifest_path = BUILD_DIR / "manifest.json"
        if manifest_path.exists():
            return FileResponse(manifest_path)
        raise HTTPException(status_code=404)
    
    # Mount static directory for CSS, JS, etc.
    static_dir = BUILD_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info(f"Mounted static files from {static_dir}")
    
    logger.info(f"Static files configured for {BUILD_DIR}")
else:
    logger.error(f"Build directory missing at {BUILD_DIR}")

# Catch-all for React SPA routes (MUST be last)
@app.get("/{full_path:path}")
async def serve_spa(full_path: str, request: Request):
    logger.info(f"Catch-all route called for: {full_path}")
    
    # Never handle API routes in catch-all
    if full_path.startswith("api/"):
        logger.warning(f"API route {full_path} reached catch-all - this shouldn't happen")
        raise HTTPException(status_code=404, detail="API endpoint not found")
    
    # Serve React app for frontend routes
    if INDEX_HTML_PATH.exists():
        try:
            with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
                content = f.read()
                content = content.replace("__BASE_URL__", BASE_URL)
                logger.info(f"Serving React index.html for {full_path}")
                return HTMLResponse(content=content)
        except Exception as e:
            logger.error(f"Error serving index.html: {str(e)}")
            raise HTTPException(status_code=500, detail="Error serving page")
    else:
        logger.error("index.html not found")
        raise HTTPException(status_code=500, detail="Frontend not built")

def signal_handler(sig, frame):
    logger.info("Received shutdown signal")
    stop_streamlit()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    
    # Start streamlit in background
    start_streamlit()
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=int(os.getenv("PORT", 8000)), 
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        stop_streamlit()
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        stop_streamlit()