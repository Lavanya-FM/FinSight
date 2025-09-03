from services.base import client, admin_client, ensure_session
from pathlib import Path
import tempfile
import hashlib
from datetime import datetime, timezone
import logging
import uuid
import json
from supabase import create_client

logger = logging.getLogger(__name__)

def calculate_file_hash(local_path: str) -> str:
    try:
        hash_sha256 = hashlib.sha256()
        with open(local_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {str(e)}")
        return ""
    
def validate_user_id(user_id: str) -> bool:
    """Validate user_id format"""
    if not isinstance(user_id, str):
        return False
    try:
        # Check if it's a valid UUID
        uuid.UUID(user_id)
        return True
    except ValueError:
        return False

def save_uploaded_file(user_id: str, file_name: str, local_path: str) -> int:
    """Alternative approach - store file content directly in database as base64"""
    try:
        # Validate user_id format
        if not isinstance(user_id, str) or len(user_id.split('-')) != 5:
            raise ValueError("Invalid user_id format; expected UUID")

        if not file_name:
            raise ValueError("File name cannot be empty")
        
        if not Path(local_path).exists():
            raise ValueError(f"File does not exist: {local_path}")
        
        # Check if user exists in database using admin client
        user_check = admin_client.table('users').select('user_id').eq('user_id', user_id).execute()
        if not user_check.data:
            raise ValueError(f"User {user_id} not found in database")

        # Read file content and compute size
        with open(local_path, 'rb') as f:
            file_content = f.read()
        
        if not file_content:
            raise ValueError("File is empty")

        file_size = len(file_content)
        
        # Calculate file hash
        file_hash = calculate_file_hash(local_path)
        if not file_hash:
            raise ValueError("Failed to calculate file hash")

        # Determine file type
        file_extension = Path(file_name).suffix[1:].lower() if Path(file_name).suffix else 'unknown'
        if not file_extension or file_extension == 'unknown':
            file_extension = 'txt'
            logger.warning(f"No file extension for {file_name}, defaulting to 'txt'")
        
        # Convert file content to base64 for database storage
        import base64
        file_content_b64 = base64.b64encode(file_content).decode('utf-8')
        
        # Create a "virtual" storage path for compatibility
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        storage_path = f"database_stored/user_{user_id}/uploads/{timestamp}_{file_name}"
        
        logger.info(f"Storing file in database: {file_name} (size: {file_size} bytes)")

        # Store everything directly in the database
        insert_data = {
            'user_id': user_id,
            'file_name': file_name,
            'file_path': storage_path,  # Virtual path
            'file_size': file_size,
            'file_type': file_extension,
            'file_hash': file_hash,
            'uploaded_at': datetime.now(timezone.utc).isoformat(),
            'file_content': file_content_b64  # Store actual file content
        }
        
        logger.info(f"Inserting file metadata and content: {file_name}")

        try:
            response = admin_client.table('uploaded_files').insert(insert_data).execute()
            
            if not response.data:
                raise ValueError("No data returned from database insert")
                
            file_id = response.data[0]['file_id']
            logger.info(f"Successfully stored file {file_name} for user {user_id} with file_id {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Database insert error: {str(e)}")
            raise ValueError(f"Failed to save file: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error saving file {file_name}: {str(e)}")
        raise
       

def download_file_content(file_path: str) -> bytes:
    """Download file content - updated for database storage"""
    try:
        if not file_path:
            raise ValueError("File path cannot be empty")
            
        logger.info(f"Retrieving file from database: {file_path}")
        
        # Check if it's a database-stored file
        if file_path.startswith("database_stored/"):
            # Get file content from database
            response = admin_client.table('uploaded_files').select('file_content').eq('file_path', file_path).execute()
            
            if not response.data:
                raise ValueError(f"File not found in database: {file_path}")
            
            file_content_b64 = response.data[0]['file_content']
            if not file_content_b64:
                raise ValueError(f"No file content found for: {file_path}")
            
            # Decode base64 content
            import base64
            file_content = base64.b64decode(file_content_b64)
            
            logger.info(f"Successfully retrieved file from database: {file_path} (size: {len(file_content)} bytes)")
            return file_content
        else:
            # Original storage system (fallback for existing files)
            response = admin_client.storage.from_('files').download(file_path)
            
            if not response:
                raise ValueError(f"File not found in storage: {file_path}")
                
            logger.info(f"Successfully downloaded file: {file_path} (size: {len(response)} bytes)")
            return response
        
    except Exception as e:
        logger.error(f"Error retrieving file {file_path}: {str(e)}")
        raise ValueError(f"Failed to retrieve file: {str(e)}")
            
def set_current_user_id(client, user_id: str):
    try:
        client.rpc('set_config', {'key': 'app.current_user_id', 'value': user_id}).execute()
    except Exception as e:
        logger.warning(f"Failed to set app.current_user_id: {e}")

def fetch_files(user_id: str) -> list:
    """Fetch files for a specific user"""
    try:
        if not validate_user_id(user_id):
            raise ValueError("Invalid user_id format; expected valid UUID")
            
        # Set the current user ID for RLS
        set_current_user_id(client, user_id)
        
        # Try to use regular client first (respects RLS)
        try:
            response = client.table('uploaded_files').select('*').eq('user_id', user_id).order('uploaded_at', desc=True).execute()
        except Exception as e:
            logger.warning(f"Regular client failed, using admin client: {e}")
            # Fall back to admin client
            response = admin_client.table('uploaded_files').select('*').eq('user_id', user_id).order('uploaded_at', desc=True).execute()
        
        files = response.data if response.data else []
        logger.info(f"Fetched {len(files)} files for user {user_id}")
        return files
        
    except Exception as e:
        logger.error(f"Error fetching files for user {user_id}: {str(e)}")
        return []

def download_file_content(file_path: str) -> bytes:
    """Download file content from storage"""
    try:
        if not file_path:
            raise ValueError("File path cannot be empty")
            
        logger.info(f"Downloading file from storage: {file_path}")
        
        # Use admin client for storage operations
        response = admin_client.storage.from_('files').download(file_path)
        
        if not response:
            raise ValueError(f"File not found in storage: {file_path}")
            
        logger.info(f"Successfully downloaded file: {file_path} (size: {len(response)} bytes)")
        return response
        
    except Exception as e:
        logger.error(f"Error downloading file {file_path}: {str(e)}")
        raise ValueError(f"Failed to download file: {str(e)}")

def delete_file(user_id: str, file_id: int) -> bool:
    """Delete a file (both from storage and database)"""
    try:
        if not validate_user_id(user_id):
            raise ValueError("Invalid user_id format; expected valid UUID")
            
        if not isinstance(file_id, int) or file_id <= 0:
            raise ValueError("Invalid file_id")
        
        # First, get file info to get storage path
        response = admin_client.table('uploaded_files').select('file_path').eq('file_id', file_id).eq('user_id', user_id).execute()
        
        if not response.data:
            raise ValueError(f"File not found: file_id={file_id}, user_id={user_id}")
        
        file_path = response.data[0]['file_path']
        
        
        
        # Delete from database
        delete_response = admin_client.table('uploaded_files').delete().eq('file_id', file_id).eq('user_id', user_id).execute()
        
        if delete_response.data:
            logger.info(f"Successfully deleted file: file_id={file_id}")
            return True
        else:
            logger.error(f"Failed to delete file from database: file_id={file_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {str(e)}")
        return False

def get_file_info(user_id: str, file_id: int) -> dict:
    """Get file information"""
    try:
        if not validate_user_id(user_id):
            raise ValueError("Invalid user_id format")
            
        response = admin_client.table('uploaded_files').select('*').eq('file_id', file_id).eq('user_id', user_id).execute()
        
        if response.data:
            return response.data[0]
        else:
            return {}
            
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        return {}