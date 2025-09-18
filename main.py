import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel

# -----------------------------
# Logging configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("="*50)
logger.info("STARTING FINSIGHT APPLICATION")
logger.info("="*50)

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if SUPABASE_URL and SUPABASE_ANON_KEY:
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    admin_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_SERVICE_KEY else None
    logger.info("Supabase clients initialized")
else:
    client = None
    admin_client = None
    logger.warning("Supabase not configured")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="FinSight API", docs_url="/api/docs")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Pydantic models
# -----------------------------
class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    password: str
    username: str

# -----------------------------
# Dependencies
# -----------------------------
def get_current_user(request: Request) -> Optional[dict]:
    """Get current authenticated user from request cookies"""
    try:
        token = request.cookies.get("supabase_token")
        if not token or not client:
            return None
        
        temp_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        temp_client.auth.set_session(token, token)
        session_response = temp_client.auth.get_session()
        
        if session_response and hasattr(session_response, "data") and session_response.data.session:
            user_id = session_response.data.session.user.id
            if admin_client:
                user_data = admin_client.table("users").select("user_id, username, role").eq("user_id", user_id).execute()
                if user_data.data and len(user_data.data) > 0:
                    return {
                        "user_id": user_data.data[0]["user_id"],
                        "username": user_data.data[0]["username"],
                        "role": user_data.data[0]["role"],
                        "session": session_response.data.session
                    }
        return None
    except Exception as e:
        logger.error(f"Get current user failed: {e}")
        return None

# -----------------------------
# Middleware for logging
# -----------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"REQUEST: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"RESPONSE: {response.status_code} for {request.method} {request.url.path}")
    return response

# -----------------------------
# React build paths
# -----------------------------
PROJECT_ROOT = Path(__file__).parent
BUILD_DIR = PROJECT_ROOT / "finsight-ui" / "build"
INDEX_HTML_PATH = BUILD_DIR / "index.html"

if BUILD_DIR.exists() and INDEX_HTML_PATH.exists():
    logger.info(f"React build directory found at {BUILD_DIR}")
else:
    logger.error(f"React build directory missing: {BUILD_DIR}")

# -----------------------------
# API endpoints
# -----------------------------
@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/api/login")
async def login(request: LoginRequest):
    if not client:
        return JSONResponse({"error": "Authentication service not available"}, status_code=503)
    try:
        response = client.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        if response.user and response.session:
            user_id = response.user.id
            if admin_client:
                user_data = admin_client.table("users").select("user_id, username, role").eq("user_id", user_id).execute()
                user_info = user_data.data[0] if user_data.data else {
                    "user_id": user_id,
                    "username": request.email.split("@")[0],
                    "role": "user"
                }
                resp = JSONResponse({
                    "authenticated": True,
                    "user": {
                        "userId": user_info["user_id"],
                        "username": user_info["username"],
                        "role": user_info["role"],
                        "email": request.email
                    },
                    "session_token": response.session.access_token
                })
                resp.set_cookie("supabase_token", response.session.access_token, httponly=True, max_age=3600*24*7)
                return resp
        return JSONResponse({"error": "Invalid credentials"}, status_code=401)
    except Exception as e:
        logger.error(f"Login error: {e}")
        return JSONResponse({"error": "Login failed"}, status_code=500)

@app.post("/api/register")
async def register(request: RegisterRequest):
    if not client:
        return JSONResponse({"error": "Authentication service not available"}, status_code=503)
    try:
        response = client.auth.sign_up({
            "email": request.email,
            "password": request.password
        })
        if response.user:
            user_id = response.user.id
            if admin_client:
                admin_client.table("users").insert({
                    "user_id": user_id,
                    "username": request.username,
                    "email": request.email,
                    "role": "user"
                }).execute()
            return JSONResponse({"message": "Registration successful", "user": {"userId": user_id, "username": request.username}})
        return JSONResponse({"error": "Registration failed"}, status_code=400)
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return JSONResponse({"error": "Registration failed"}, status_code=500)

@app.get("/api/check-session")
async def check_session(current_user: Optional[dict] = Depends(get_current_user)):
    if current_user:
        return {"authenticated": True, "user": current_user}
    return {"authenticated": False, "user": None}

@app.post("/api/logout")
async def logout(request: Request):
    resp = JSONResponse({"status": "success"})
    resp.delete_cookie("supabase_token", path="/")
    return resp

# -----------------------------
# Serve React frontend
# -----------------------------
@app.get("/{full_path:path}")
async def serve_react(full_path: str):
    """Serve React app for any non-API routes"""
    if full_path.startswith("api"):
        raise HTTPException(status_code=404, detail="API route not found")
    if INDEX_HTML_PATH.exists():
        return FileResponse(INDEX_HTML_PATH)
    raise HTTPException(status_code=404, detail="Frontend not found")

# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_ANON_KEY),
        "build_exists": BUILD_DIR.exists()
    }

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
