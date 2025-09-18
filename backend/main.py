from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging
import random
import time
import re
import hashlib
from supabase import create_client, Client
from datetime import datetime
import os
from dotenv import load_dotenv
from fastapi.security import OAuth2AuthorizationCodeBearer
from jwt import decode as jwt_decode, PyJWTError
from services.inference import load_models, run_inference

# Load environment variables
load_dotenv()
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET") or os.getenv("SUPABASE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")  # Fallback for local dev

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()
app.router.redirect_slashes = False

# Validate Supabase configuration
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Supabase configuration missing: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    raise ValueError("Supabase configuration missing")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Authentication
oauth2_scheme = OAuth2AuthorizationCodeBearer(authorizationUrl="auth", tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        # Decode JWT with audience validation disabled or set to "authenticated"
        payload = jwt_decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated")
        user_id: str = payload.get("sub")
        if user_id is None:
            logger.error("JWT token missing 'sub' claim")
            raise HTTPException(status_code=401, detail="Invalid token: missing user ID")
        logger.info(f"Authenticated user: {user_id}")
        return user_id
    except PyJWTError as e:
        logger.error(f"JWT decode error: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Not authenticated: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in get_current_user: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
          
@app.get("/api/v1/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "ok"}

@app.get("/api/v1/analysis")
async def get_analysis():
    logger.info("Analysis endpoint called")
    return {"message": "Analysis data not implemented"}

@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Fetch all reports
@app.get("/api/v1/reports")
async def get_reports(current_user: str = Depends(get_current_user)):
    logger.info(f"Fetching reports for user: {current_user}")
    try:
        response = supabase.table("reports").select("*").eq("user_id", current_user).execute()
        if not response.data:
            logger.info(f"No reports found for user: {current_user}")
            return []  # Return empty array if no reports
        reports = [{
            "report_id": report["report_id"],
            "file_id": report["file_id"],
            "user_id": str(report["user_id"]),
            "report_details": report["report_details"],
            "action": report["action"],
            "reason": report["reason"],
            "confidence": report["confidence"],
            "generated_at": report["generated_at"],
            "updated_at": report.get("updated_at", report["generated_at"])
        } for report in response.data]
        logger.info(f"Fetched {len(reports)} reports for user: {current_user}")
        return reports
    except Exception as e:
        logger.error(f"Error fetching reports: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch reports: {str(e)}")
      
# Save report
class SaveReportRequest(BaseModel):
    file_id: int
    user_id: str
    report_details: dict
    action: str
    reason: str
    confidence: float

@app.post("/api/v1/save-report")
async def save_report(req: SaveReportRequest, current_user: str = Depends(get_current_user)):
    if req.user_id != current_user:
        raise HTTPException(status_code=403, detail="Unauthorized to save report for this user")
    logger.info(f"Saving report for file_id: {req.file_id}, user_id: {req.user_id}")
    try:
        report_data = {
            "file_id": req.file_id,
            "user_id": req.user_id,
            "report_details": req.report_details,
            "action": req.action,
            "reason": req.reason,
            "confidence": req.confidence,
            "generated_at": datetime.utcnow().isoformat(),
        }
        response = supabase.table("reports").insert(report_data).execute()
        logger.info(f"Report saved with ID: {response.data[0]['report_id']}")
        return {
            "report_id": response.data[0]["report_id"],
            **report_data,
            "updated_at": response.data[0].get("updated_at", report_data["generated_at"])
        }
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        raise HTTPException(status_code=500, detail="Failed to save report")
    
# Store uploaded file metadata
async def store_file_metadata(file: UploadFile, user_id: str, cibil_score: int):
    try:
        # Read file content for hashing
        file_content = await file.read()
        file_hash = hashlib.sha256(file_content).hexdigest()
        # Reset file pointer
        await file.seek(0)
        
        # Generate a unique file path (e.g., using user_id and filename)
        file_path = f"uploads/{user_id}/{file.filename}"
        
        file_data = {
            "file_name": file.filename,
            "file_path": file_path,
            "file_size": file.size,
            "file_type": file.content_type,
            "user_id": user_id,
            "uploaded_at": datetime.utcnow().isoformat(),
            "file_hash": file_hash,
            "file_content": None,  # Store as None unless you need to store content
            "cibil_score": cibil_score if 300 <= cibil_score <= 900 else None
        }
        response = supabase.table("uploaded_files").insert(file_data).execute()
        return response.data[0]["file_id"]
    except Exception as e:
        logger.error(f"Error storing file metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to store file metadata")   
try:
    models = load_models("backend/models")
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {e}")

class PredictionRequest(BaseModel):
    model_name: str
    data: dict

@app.delete("/api/v1/reports/{report_id}")
async def delete_report(report_id: int, current_user: str = Depends(get_current_user)):
    logger.info(f"Deleting report {report_id} for user {current_user}")
    try:
        response = supabase.table("reports").delete().eq("report_id", report_id).eq("user_id", current_user).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Report not found or not authorized")
        logger.info(f"Report {report_id} deleted successfully")
        return {"status": "success", "message": f"Report {report_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting report: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete report")
    
@app.post("/api/v1/predict")
async def predict(req: PredictionRequest):
    if req.model_name not in models:
        logger.error(f"Model {req.model_name} not found")
        raise HTTPException(status_code=404, detail=f"Model {req.model_name} not found")
    prediction = run_inference(models[req.model_name], req.data)
    logger.info(f"Prediction made for model {req.model_name}")
    return {"model": req.model_name, "prediction": prediction}

def extract_financial_data(file: UploadFile, cibil_score: int):
    """Mock function to simulate extracting financial data from a document."""
    start_time = time.time()
    logger.info(f"Extracting financial data from file: {file.filename}")

    # Simulate document processing
    time.sleep(0.5)  # Mimic processing delay

    # Mock applicant name extraction from filename
    filename = file.filename.lower()
    applicant_name = re.sub(r'[_\-\.]', ' ', filename.split('.')[0]).title() or 'Unknown Applicant'

    # Generate dynamic financial metrics based on CIBIL score
    monthly_income = random.randint(40000, 120000)
    monthly_expenses = int(monthly_income * random.uniform(0.5, 0.8))
    savings_rate = round((monthly_income - monthly_expenses) / monthly_income * 100, 1)
    average_balance = random.randint(50000, 200000)
    debt_to_income_ratio = random.randint(20, 40)
    total_transactions = random.randint(40, 150)
    income_variability = 'Low' if cibil_score > 700 else 'Moderate' if cibil_score > 600 else 'High'

    # Risk assessment
    overall_risk_score = min(95, cibil_score // 10 + random.randint(5, 15))
    risk_category = 'Low Risk' if overall_risk_score > 70 else 'Medium Risk' if overall_risk_score > 50 else 'High Risk'
    risk_factors = [
        {"factor": "Income Stability", "score": min(100, overall_risk_score + random.randint(-5, 5))},
        {"factor": "Debt Load", "score": min(100, overall_risk_score + random.randint(-10, 10))},
        {"factor": "Transaction Consistency", "score": min(100, overall_risk_score + random.randint(-5, 5))},
    ]

    # Decision summary
    final_decision = 'APPROVED' if cibil_score > 650 else 'REJECTED'
    reason = (
        "Stable income and good credit score" if final_decision == 'APPROVED'
        else "Insufficient credit score or high risk"
    )
    confidence_score = str(round(random.uniform(0.85, 0.95), 2)) if final_decision == 'APPROVED' else str(round(random.uniform(0.60, 0.80), 2))
    recommended_loan_amount = f"₹{random.randint(300000, 1000000)}" if final_decision == 'APPROVED' else "₹0"
    interest_rate_bracket = "8-10%" if cibil_score > 750 else "10-12%" if cibil_score > 650 else "N/A"

    # Chart data
    monthly_trends = [
        {"month": "Jan", "income": monthly_income, "expenses": monthly_expenses, "balance": monthly_income - monthly_expenses},
        {"month": "Feb", "income": int(monthly_income * random.uniform(0.95, 1.05)), "expenses": int(monthly_expenses * random.uniform(0.95, 1.05)), "balance": monthly_income - monthly_expenses},
        {"month": "Mar", "income": int(monthly_income * random.uniform(0.95, 1.05)), "expenses": int(monthly_expenses * random.uniform(0.95, 1.05)), "balance": monthly_income - monthly_expenses},
    ]
    category_breakdown = [
        {"name": "Housing", "value": int(monthly_expenses * 0.4), "color": "#0088FE"},
        {"name": "Food", "value": int(monthly_expenses * 0.2), "color": "#00C49F"},
        {"name": "Transport", "value": int(monthly_expenses * 0.15), "color": "#FFBB28"},
        {"name": "Others", "value": int(monthly_expenses * 0.25), "color": "#FF8042"},
    ]
    transaction_volume = [
        {"month": "Jan", "volume": total_transactions},
        {"month": "Feb", "volume": int(total_transactions * random.uniform(0.9, 1.1))},
        {"month": "Mar", "volume": int(total_transactions * random.uniform(0.9, 1.1))},
    ]

    # Detailed analysis
    detailed_analysis = {
        "account_stability": "Stable" if cibil_score > 700 else "Moderate",
        "income_consistency": "High" if income_variability == "Low" else "Moderate",
        "expense_patterns": "Regular",
        "banking_behavior": "Conservative" if savings_rate > 30 else "Balanced",
        "seasonal_variations": "Minimal",
        "transaction_frequency": "Moderate",
        "cash_flow_pattern": "Positive" if savings_rate > 20 else "Neutral",
    }

    # Recommendations
    recommendations = [
        "Maintain current savings rate" if savings_rate > 20 else "Increase savings to improve financial stability",
        "Consider reducing discretionary spending",
        "Monitor CIBIL score regularly for updates",
    ]

    processing_time = f"{(time.time() - start_time):.1f}s"

    return {
        "financial_metrics": {
            "monthly_income": monthly_income,
            "cibil_score": cibil_score,
            "savings_rate": savings_rate,
            "average_balance": average_balance,
            "monthly_expenses": monthly_expenses,
            "debt_to_income_ratio": debt_to_income_ratio,
            "total_transactions": total_transactions,
            "income_variability": income_variability,
        },
        "decision_summary": {
            "final_decision": final_decision,
            "reason": reason,
            "confidence_score": confidence_score,
            "recommended_loan_amount": recommended_loan_amount,
            "interest_rate_bracket": interest_rate_bracket,
        },
        "risk_assessment": {
            "overall_risk_score": overall_risk_score,
            "stability_score": risk_factors[0]["score"],
            "risk_category": risk_category,
            "risk_factors": risk_factors,
        },
        "compliance_checks": {
            "aml_status": "Cleared",
            "fraud_indicators": 0,
            "document_authenticity": "Verified",
            "identity_verification": "Verified",
            "address_verification": "Verified",
        },
        "detailed_analysis": detailed_analysis,
        "chart_data": {
            "monthly_trends": monthly_trends,
            "category_breakdown": category_breakdown,
            "transaction_volume": transaction_volume,
        },
        "recommendations": recommendations,
        "file_analysis": {
            "file_name": file.filename,
            "file_size": f"{(file.size / 1024 / 1024):.1f}MB",
            "processing_time": processing_time,
            "applicant_name": applicant_name,
        },
        "analysis_version": "1.0.0",
    }

@app.post("/api/v1/analyze-document")
async def analyze_bank_statement(
    cibil_score: int = Form(...),
    files: List[UploadFile] = File(...),
    analysis_type: str = Form(...),
    include_charts: str = Form(...),
    current_user: str = Depends(get_current_user)
):
    logger.info(f"Analyze endpoint called with cibil_score: {cibil_score}, files: {[file.filename for file in files]}, analysis_type: {analysis_type}, include_charts: {include_charts}")
    
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    
    # Validate file types
    allowed_types = ['application/pdf', 'text/csv', 'application/vnd.ms-excel', 
                     'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'text/plain']
    for file in files:
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Unsupported file type for {file.filename}: {file.content_type}")

    # Store file metadata and get file_id
    file_id = await store_file_metadata(files[0], current_user, cibil_score)

    # Process the first file for simplicity (extend for multiple files if needed)
    analysis_result = extract_financial_data(files[0], cibil_score)
    
    # Add file_id to file_analysis
    analysis_result["file_analysis"]["file_id"] = file_id

    return {
        "cibil_score": cibil_score,
        "files_received": [file.filename for file in files],
        **analysis_result
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)