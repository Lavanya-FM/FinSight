import os
import logging
import signal
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from dotenv import load_dotenv
from typing import Optional
from pathlib import Path
import time
from fastapi.staticfiles import StaticFiles


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

BASE_URL = os.getenv("BASE_URL")

# Create FastAPI app with explicit prefix handling
app = FastAPI(
    title="FinSight API",
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
        return response
    except Exception as e:
        logger.error(f"Failed to sign out from Supabase: {str(e)}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    try:
        return {
            "status": "healthy",
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
        "base_url": BASE_URL
    }

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
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
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
    except Exception as e:
        logger.error(f"Server error: {str(e)}")