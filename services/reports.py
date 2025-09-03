# services/reports.py
from datetime import datetime
import logging
import json
import uuid
import hashlib
import time
from pathlib import Path
from services.base import client, admin_client

logger = logging.getLogger(__name__)

def validate_user_id(user_id: str) -> bool:
    """Validate user_id format"""
    if not isinstance(user_id, str):
        return False
    try:
        uuid.UUID(user_id)
        return True
    except ValueError:
        return False

def validate_action(action: str) -> bool:
    """Validate action is one of the allowed values"""
    valid_actions = ['Approve', 'Reject', 'Pending', 'Review', 'Unknown']
    return action in valid_actions

def save_report(user_id: str, file_id: int, decision: dict) -> int:
    """
    Save a report to the database without storing files.
    Returns the report_id if successful, raises an exception otherwise.
    """
    try:
        # Input validation
        if not validate_user_id(user_id):
            raise ValueError("Invalid user_id format; expected valid UUID")
            
        if not isinstance(file_id, int) or file_id <= 0:
            raise ValueError("file_id must be a positive integer")
            
        if not decision:
            raise ValueError("decision cannot be empty")
    
        # Extract and validate decision components
        action = decision.get("Action", "Unknown")
        reason = decision.get("Reason", "")
        confidence = decision.get("Confidence")
        
        if not validate_action(action):
            logger.warning(f"Invalid action '{action}', defaulting to 'Unknown'")
            action = "Unknown"
        
        if confidence is not None:
            try:
                confidence = float(confidence)
                if confidence < 0 or confidence > 100:
                    logger.warning(f"Confidence {confidence} out of range, setting to None")
                    confidence = None
            except (ValueError, TypeError):
                logger.warning(f"Invalid confidence value '{confidence}', setting to None")
                confidence = None

        # Prepare report data
        insert_data = {
            'user_id': user_id,
            'file_id': file_id,
            'action': action,
            'reason': reason,
            'generated_at': datetime.now().isoformat(),
            'report_details': decision.get("report_details", {})
        }
        
        if confidence is not None:
            insert_data['confidence'] = confidence
            
        logger.debug(f"Report data: {insert_data}")
        
        # Insert report with retry logic
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = admin_client.table('reports').insert(insert_data).execute()
                
                if response.data and len(response.data) > 0:
                    report_id = response.data[0].get('report_id')
                    if report_id:
                        logger.info(f"Report saved successfully with ID: {report_id} (attempt {attempt + 1})")
                        return report_id
                    else:
                        raise ValueError("Report inserted but no report_id returned")
                else:
                    raise ValueError("No data returned from insert operation")
                    
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"Database error (attempt {attempt + 1}): {str(e)}. Retrying...")
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"Database operation failed after {max_attempts} attempts: {str(e)}")
                    raise e
        
    except Exception as e:
        error_msg = f"Error saving report for user {user_id}, file {file_id}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
def fetch_reports(user_id: str = None) -> list:
    """
    Fetch reports with updated schema (separate action, reason, confidence columns)
    """
    try:
        if user_id:
            if not validate_user_id(user_id):
                raise ValueError("Invalid user_id format")
            
            # Use admin client to avoid RLS issues
            response = admin_client.table("reports")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("generated_at", desc=True)\
                .execute()
        else:
            # Fetch all reports (admin view)
            response = admin_client.table("reports")\
                .select("*")\
                .order("generated_at", desc=True)\
                .execute()
        
        reports = []
        if response.data:
            for report in response.data:
                
                # Create decision dict for backward compatibility
                report['decision'] = {
                    "Action": report.get('action', 'Unknown'),
                    "Reason": report.get('reason', ''),
                    "Confidence": report.get('confidence')
                }
                
                reports.append(report)
            
            logger.info(f"Fetched {len(reports)} reports" + (f" for user {user_id}" if user_id else ""))
        else:
            logger.info(f"No reports found" + (f" for user {user_id}" if user_id else ""))
        
        return reports
        
    except Exception as e:
        logger.error(f"Error fetching reports: {str(e)}")
        return []

def fetch_report_by_id(report_id: int, user_id: str = None):
    """
    Fetch a specific report by its ID - simplified without audit history
    """
    try:
        if not isinstance(report_id, int) or report_id <= 0:
            raise ValueError("Invalid report_id")
        
        # Use basic reports table instead of reports_with_audit view
        query = admin_client.table("reports").select("*").eq("report_id", report_id)
        
        # Add user filter if provided
        if user_id:
            if not validate_user_id(user_id):
                raise ValueError("Invalid user_id format")
            query = query.eq("user_id", user_id)
        
        response = query.execute()
        
        if response.data:
            report = response.data[0]
            
            
            # Create decision dict for backward compatibility
            report['decision'] = {
                "Action": report.get('action', 'Unknown'),
                "Reason": report.get('reason', ''),
                "Confidence": report.get('confidence')
            }
            
            logger.info(f"Fetched report: report_id={report_id}")
            return report
        else:
            logger.warning(f"Report not found: report_id={report_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching report {report_id}: {str(e)}")
        return None

def update_report_decision(report_id: int, new_action: str, admin_id: str, reason: str = "", confidence: float = None) -> bool:
    """
    Update report decision with separate columns - simplified without automatic audit logging
    """
    try:
        # Validate inputs
        if not isinstance(report_id, int) or report_id <= 0:
            raise ValueError("report_id must be a positive integer")
        if not new_action:
            raise ValueError("new_action cannot be empty")
        if not validate_action(new_action):
            raise ValueError(f"Invalid action: {new_action}")
        if not admin_id:
            raise ValueError("admin_id cannot be empty")
        
        # Validate confidence if provided
        if confidence is not None:
            try:
                confidence = float(confidence)
                if confidence < 0 or confidence > 100:
                    raise ValueError("Confidence must be between 0 and 100")
            except (ValueError, TypeError):
                raise ValueError("Invalid confidence value")
        
        # Prepare update data
        update_data = {
            'action': new_action,
            'reason': reason,
            'confidence': confidence,
            'updated_at': datetime.now().isoformat(),
            'updated_by': admin_id
        }
        
        # Update the report
        update_response = admin_client.table('reports')\
            .update(update_data)\
            .eq('report_id', report_id)\
            .execute()
        
        if not update_response.data:
            raise ValueError("Failed to update report - no data returned")
        
        logger.info(f"Report {report_id} decision updated to '{new_action}' by admin {admin_id}")
        return True
        
    except Exception as e:
        error_msg = f"Error updating report {report_id}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_audit_history(report_id: int, user_id: str = None) -> list:
    """
    Get audit history for a specific report - simplified to work with current schema
    """
    try:
        if not isinstance(report_id, int) or report_id <= 0:
            raise ValueError("Invalid report_id")
        
        # Basic query without user join since it may not exist
        query = admin_client.table("audit_log")\
            .select("*")\
            .eq("report_id", report_id)\
            .order("changed_at", desc=True)
        
        response = query.execute()
        
        audit_logs = response.data if response.data else []
        logger.info(f"Fetched {len(audit_logs)} audit entries for report {report_id}")
        return audit_logs
        
    except Exception as e:
        logger.error(f"Error fetching audit history for report {report_id}: {str(e)}")
        return []

def delete_report(report_id: int, user_id: str = None) -> bool:
    """
    Delete a report (soft delete by marking as deleted)
    """
    try:
        if not isinstance(report_id, int) or report_id <= 0:
            raise ValueError("Invalid report_id")
        
        update_data = {
            "is_deleted": True,
            "deleted_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        query = admin_client.table("reports").update(update_data).eq("report_id", report_id)
        
        # Add user ownership check if provided
        if user_id:
            if not validate_user_id(user_id):
                raise ValueError("Invalid user_id format")
            query = query.eq("user_id", user_id)
        
        response = query.execute()
        
        if response.data:
            logger.info(f"Report deleted: report_id={report_id}, user_id={user_id}")
            return True
        else:
            logger.error(f"Failed to delete report: report_id={report_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting report: {str(e)}")
        return False

def get_user_report_statistics(user_id: str):
    """
    Get report statistics for a specific user with updated schema
    """
    try:
        if not validate_user_id(user_id):
            raise ValueError("Invalid user_id format")
        
        reports = fetch_reports(user_id)
        
        stats = {
            "total_reports": len(reports),
            "approved": 0,
            "rejected": 0,
            "pending": 0,
            "under_review": 0,
            "recent_reports": reports[:5]
        }
        
        # Count by action
        for report in reports:
            action = report.get('action', 'Unknown').lower()
            if action == 'approve':
                stats["approved"] += 1
            elif action == 'reject':
                stats["rejected"] += 1
            elif action == 'pending':
                stats["pending"] += 1
            elif action == 'review':
                stats["under_review"] += 1
        
        # Calculate approval rate
        total_decided = stats["approved"] + stats["rejected"]
        if total_decided > 0:
            stats["approval_rate"] = (stats["approved"] / total_decided) * 100
        else:
            stats["approval_rate"] = 0
        
        logger.info(f"Generated statistics for user {user_id}: {stats['total_reports']} reports")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting user statistics: {str(e)}")
        return {
            "total_reports": 0,
            "approved": 0,
            "rejected": 0,
            "pending": 0,
            "under_review": 0,
            "approval_rate": 0,
            "recent_reports": []
        }

def get_system_report_statistics():
    """
    Get system-wide report statistics with updated schema
    """
    try:
        reports = fetch_reports()  # Fetch all reports
        
        if not reports:
            return {
                "total_reports": 0,
                "total_users_with_reports": 0,
                "approved": 0,
                "rejected": 0,
                "pending": 0,
                "under_review": 0,
                "approval_rate": 0,
                "avg_reports_per_user": 0
            }
        
        unique_users = len(set(r["user_id"] for r in reports))
        
        stats = {
            "total_reports": len(reports),
            "total_users_with_reports": unique_users,
            "approved": 0,
            "rejected": 0,
            "pending": 0,
            "under_review": 0,
        }
        
        # Count by action
        for report in reports:
            action = report.get('action', 'Unknown').lower()
            if action == 'approve':
                stats["approved"] += 1
            elif action == 'reject':
                stats["rejected"] += 1
            elif action == 'pending':
                stats["pending"] += 1
            elif action == 'review':
                stats["under_review"] += 1
        
        # Calculate rates
        total_decided = stats["approved"] + stats["rejected"]
        if total_decided > 0:
            stats["approval_rate"] = (stats["approved"] / total_decided) * 100
        else:
            stats["approval_rate"] = 0
            
        if unique_users > 0:
            stats["avg_reports_per_user"] = stats["total_reports"] / unique_users
        else:
            stats["avg_reports_per_user"] = 0
        
        logger.info(f"Generated system statistics: {stats['total_reports']} reports across {unique_users} users")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting system statistics: {str(e)}")
        return {
            "total_reports": 0,
            "total_users_with_reports": 0,
            "approved": 0,
            "rejected": 0,
            "pending": 0,
            "under_review": 0,
            "approval_rate": 0,
            "avg_reports_per_user": 0
        }