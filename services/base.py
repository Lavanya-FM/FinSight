# services/base.py
import os
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env file from project root
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

# Environment variables
url: str = os.environ.get("SUPABASE_URL")
anon_key: str = os.environ.get("SUPABASE_ANON_KEY")
service_key: str = os.environ.get("SUPABASE_SERVICE_KEY")

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
    SUPABASE_SERVICE_KEY = st.secrets["SUPABASE_SERVICE_KEY"]
except KeyError:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not all([SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY]):
    raise ValueError("Missing required environment variables")
# Clients
logger.info(f"Initializing Supabase client with URL: {url[:10]}... and anon_key: {anon_key[:4]}...")

client: Client = create_client(url, anon_key)
admin_client: Client = create_client(url, service_key)

admin_client: Client = None
if service_key:
    logger.info("Initializing Supabase admin client (service role key).")
    admin_client = create_client(url, service_key)
else:
    logger.warning("No SUPABASE_SERVICE_KEY found. Admin operations will be disabled.")

#def ensure_session():
#    """Ensure session exists (but don't require it for service operations)"""
#    try:
#        session = client.auth.get_session()
#        if session:
#            logger.info("Active session found")
#        else:
#            logger.warning("No session tokens found in st.session_state")
#    except Exception as e:
#        logger.warning(f"Session check failed: {e}")

def ensure_session(user_id: Optional[str] = None) -> bool:
    """
    Ensure user session is valid
    """
    try:
         #Check if user is already authenticated
        user = client.auth.get_user()
        if user and user.user:
            logger.info(f"User session active: {user.user.id}")
            return True
        
        # If user_id provided but no session, we can't establish one without credentials
        if user_id:
            logger.warning(f"No active session for user {user_id}, using admin client")
            return False
        if "access_token" in st.session_state and st.session_state["access_token"]:
            # Set the session for the user client
            client.auth.set_session(st.session_state["access_token"], st.session_state.get("refresh_token"))
            return True
        return False
    except Exception as e:
        logger.error(f"Session validation error: {e}")
        return False

def get_current_user():
    """
    Get the current authenticated user
    """
    try:
        user = client.auth.get_user()
        return user.user if user else None
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        return None
    
# Function to ensure the 'files' bucket exists
def ensure_bucket():
    """Ensure the files bucket exists"""
    try:
        admin_client.storage.get_bucket('files')
        logger.info("Ensured 'files' bucket exists")
    except Exception as e:
        logger.info("Creating 'files' bucket")
        admin_client.storage.create_bucket('files', {'public': True})


# Test connection
try:
    client.table("users").select("*", count="exact").limit(1).execute()
    logger.info("✅ Supabase client connection successful")
except Exception as e:
    logger.error(f"❌ Supabase client connection failed: {str(e)}")
    raise

# Ensure bucket exists on load
ensure_bucket()

# Export
__all__ = ["client", "admin_client", "ensure_bucket"]

'''
Two clients:

client → uses anon_key (for normal app usage).

admin_client → uses service_key (for admin tasks like bucket creation).
Bucket operations prefer admin_client (service role key).

Graceful fallback: if service key not present, it still works but may not be able to create/update buckets.
'''