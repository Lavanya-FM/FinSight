from services.base import client
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import logging
import re
from typing import Tuple, Optional
import json
import streamlit as st

# Setup logging
logger = logging.getLogger(__name__)

def register_user(user_name: str, email: str, password: str, role: str = 'user') -> str:
    try:
        # Validate
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            raise ValueError("Invalid email format")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        # Check if user already exists
        existing_user = client.table("users").select("email").eq("email", email.lower().strip()).execute()
        if existing_user.data:
            raise ValueError("User with this email already exists")

        # Hash password
        password_hash = generate_password_hash(password)

        # Insert into users table
        response = client.table("users").insert({
            "username": user_name,
            "email": email.lower().strip(),
            "password_hash": password_hash,
            "role": role,
            "created_at": datetime.utcnow().isoformat(),
            "last_login": datetime.utcnow().isoformat()
        }).execute()

        if response.data:
            user_id = response.data[0]["user_id"]
            logger.info(f"✅ Registered user {email} with ID {user_id}")
            return user_id
        else:
            raise ValueError("Failed to insert user into DB")
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise

def login_user(email: str, password: str):
    """
    Login using custom users table (not Supabase auth.users)
    """
    try:
        email = email.lower().strip()
        logger.info(f"🔍 Attempting login for: {email}")
        
        # Fetch user from custom users table
        response = client.table("users").select("*").eq("email", email).execute()
        
        if not response.data:
            logger.warning(f"❌ User not found in users table: {email}")
            return None
        
        user = response.data[0]  # Get first user (should be unique by email)
        logger.info(f"🔍 Found user: {user['username']} (ID: {user['user_id']})")
        
        # Verify password
        if check_password_hash(user["password_hash"], password):
            # Update last login
            client.table("users").update({
                "last_login": datetime.utcnow().isoformat()
            }).eq("user_id", user["user_id"]).execute()

            logger.info(f"✅ {email} logged in successfully")
            
            # Store session info in streamlit session state
            st.session_state["user_id"] = user["user_id"]
            st.session_state["email"] = user["email"]
            st.session_state["username"] = user["username"]
            st.session_state["role"] = user["role"]
            st.session_state["logged_in"] = True
            
            return (user["user_id"], user["username"], user["role"])
        else:
            logger.warning(f"❌ Incorrect password for {email}")
            return None
            
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        return None

def logout_user():
    """Clear session state"""
    try:
        # Clear all session state
        for key in ["user_id", "email", "username", "role", "logged_in", "access_token", "refresh_token"]:
            if key in st.session_state:
                del st.session_state[key]
        logger.info("✅ User logged out successfully")
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")

def get_user_by_email(email: str):
    """Get user details by email"""
    try:
        response = client.table("users").select("*").eq("email", email.lower().strip()).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error fetching user: {str(e)}")
        return None

def get_user_by_id(user_id: str):
    """Get user details by user_id"""
    try:
        response = client.table("users").select("*").eq("user_id", user_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error fetching user by ID: {str(e)}")
        return None

# Optional: If you want to use Supabase auth.users instead of custom table
def register_user_with_supabase_auth(user_name: str, email: str, password: str, role: str = 'user'):
    """
    Alternative: Register using Supabase's built-in authentication
    This creates user in auth.users table automatically
    """
    try:
        # Validate
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            raise ValueError("Invalid email format")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        # Register with Supabase Auth
        auth_response = client.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "username": user_name,
                    "role": role
                }
            }
        })

        if auth_response.user:
            # Also add to custom users table for additional data
            client.table("users").insert({
                "user_id": auth_response.user.id,
                "username": user_name,
                "email": email,
                "role": role,
                "created_at": datetime.utcnow().isoformat(),
                "last_login": datetime.utcnow().isoformat()
            }).execute()
            
            logger.info(f"✅ Registered user {email} with Supabase Auth")
            return auth_response.user.id
        else:
            raise ValueError("Failed to register with Supabase Auth")
            
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise

def login_user_with_supabase_auth(email: str, password: str):
    """
    Alternative: Login using Supabase's built-in authentication
    """
    try:
        auth_response = client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if auth_response.user and auth_response.session:
            # Store session info
            st.session_state["access_token"] = auth_response.session.access_token
            st.session_state["refresh_token"] = auth_response.session.refresh_token
            st.session_state["user_id"] = auth_response.user.id
            st.session_state["email"] = auth_response.user.email
            st.session_state["logged_in"] = True
            
            # Get additional user data from custom table
            user_data = get_user_by_id(auth_response.user.id)
            if user_data:
                st.session_state["username"] = user_data["username"]
                st.session_state["role"] = user_data["role"]
            
            logger.info(f"✅ {email} logged in successfully with Supabase Auth")
            return (auth_response.user.id, user_data["username"] if user_data else email, user_data["role"] if user_data else "user")
        else:
            logger.warning(f"❌ Login failed for {email}")
            return None
            
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        return None