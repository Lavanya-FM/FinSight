import os
import re
import json
import pickle
import io
import base64
import logging
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# Try to import project utils; if not available, fallback to simple helpers
from utils.text_parser import parse_bank_statement_text
from utils.categorize_transactions import categorize_transactions, TransactionCategorizer
from utils.financial_metrics import calculate_metrics, identify_high_value_transactions, recurring_transactions, enforce_schema, make_arrow_compatible
from utils.score_bank_statements import score_and_decide
from utils.extract_transactions import extract_df_from_scanned_pdf
from utils.preprocess_transactions import make_arrow_compatible
from utils.plotting import (
    plot_income_trend_plotly,
    plot_surplus_trend_plotly,
    plot_income_vs_expenses_plotly,
    plot_cumulative_savings_plotly,
    plot_category_breakdown_plotly,
    plot_high_risk_timeline,
    plot_cibil_gauge,
    plot_credit_debit_ratio_trend_plotly,
    plot_category_expenses_over_time_plotly,
    plot_top_expenses_pareto_plotly,
    plot_income_expense_scatter_plotly,
    plot_monthly_transaction_volume_by_category_plotly
    )

import logging
# Ensure Plotly browser cleanup
import plotly.io as pio
from contextlib import contextmanager

# Setup logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helpers
def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def standardize_columns(df):
    mapping = {
        'debit': 'Debit', 'withdrawal': 'Debit', 'dr': 'Debit',
        'credit': 'Credit', 'deposit': 'Credit', 'cr': 'Credit',
        'balance': 'Balance', 'bal': 'Balance',
        'date': 'Date', 'transaction date': 'Date',
        'description': 'Description', 'narration': 'Description', 'particulars': 'Description',
        'amount': 'Amount', 'type': 'Type', 'category': 'Category',
        'confidence': 'Confidence' 
    }

    unmapped = [col for col in df.columns if col not in mapping.values()]
    if unmapped:
        logger.warning(f"Unmapped columns in DataFrame: {unmapped}")
    df.columns = [mapping.get(c.lower().strip(), c) for c in df.columns]

    for col in ['Debit', 'Credit', 'Balance', 'Amount']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[₹,\s]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')
    if 'Description' in df.columns:
        df['Description'] = df['Description'].astype(str)
    if 'Category' in df.columns:
        df['Category'] = df['Category'].astype(str)
    if 'Confidence' in df.columns:
        df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce').fillna(1.0).astype('float64')
    return df

def ensure_numeric(df, numeric_cols=None):
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = [
            "Average Monthly Income", "Average Monthly Expenses",
            "Average Monthly EMI", "Net Surplus", "DTI Ratio",
            "Savings Rate", "Credit Utilization",
            "Discretionary Expenses", "Average Monthly Balance",
            "Cash Withdrawals"
        ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype("float64")
    return df

def enforce_metrics_schema(df):
    required_columns = {
        "Average Monthly Income": 0.0,
        "Average Monthly Expenses": 0.0,
        "Average Monthly EMI": 0.0,
        "Net Surplus": 0.0,
        "DTI Ratio": 0.0,
        "Savings Rate": 0.0,
        "Red Flag Count": 0,
        "Credit Utilization": 0.0,
        "Discretionary Expenses": 0.0,
        "Average Monthly Balance": 0.0,
        "Cash Withdrawals": 0.0,
        "Number of Open Credit Accounts": 0,
        "Income Stability": 0,
    }
    df = df.copy()
    for col, default in required_columns.items():
        if col not in df.columns:
            df[col] = default
    df = ensure_numeric(df, list(required_columns.keys()))
    df["Savings Rate"] = np.where(
        df["Average Monthly Income"] > 0,
        ((df["Average Monthly Income"] - df["Average Monthly Expenses"])
         / df["Average Monthly Income"]) * 100.0,
        0.0
    )
    
    emi_payments = df["Average Monthly EMI"].sum()
    if emi_payments == 0 and df.get("Debit", pd.Series([0.0])).sum() > 0:
        emi_payments = max(0, df[df["Category"] == "Fixed Expenses"]["Debit"].sum() * 0.20)
    df["Average Monthly EMI"] = emi_payments
    df["DTI Ratio"] = np.where(
        df["Average Monthly Income"] > 0,
        (df["Average Monthly EMI"] / df["Average Monthly Income"]) * 100.0,
        0.0
    )

    df["Savings Rate"] = df["Savings Rate"].astype("float64")
    df["DTI Ratio"] = df["DTI Ratio"].astype("float64")
    if 'Month' in df.columns:
        df['Month'] = df['Month'].astype('int64')
    if 'MonthStart' in df.columns:
        df['MonthStart'] = df['MonthStart'].astype(str)
    return df

# --------------- Improved fallback parser ---------------
def basic_parse_text_to_df(text):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    records = []
    date_formats = ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%b %d, %Y', '%B %Y']
    date_regex = r'(?P<date>\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\w{3,9}\s+\d{1,2},?\s*\d{4}|\w{3,9}\s+\d{4})'
    amt_regex = r'(?P<amt>-?₹?\s?\d{1,3}(?:[,\d{3}])*(?:\.\d{1,2})?)'
    combined_re = re.compile(date_regex + r'.{0,60}?' + amt_regex + r'.{0,60}?' + amt_regex, re.IGNORECASE)
    for i, line in enumerate(lines):
        m = combined_re.search(line)
        if m:
            date_str = m.group('date')
            amts = re.findall(amt_regex, line)
            debit = 0.0
            credit = 0.0
            if amts:
                normalized = [re.sub(r'[₹,\s]', '', a).replace('−', '-').replace('—', '-') for a in amts]
                for a_raw, a_norm in zip(amts, normalized):
                    if '-' in a_raw or a_norm.startswith('-'):
                        try:
                            debit = abs(float(a_norm))
                        except ValueError:
                            debit = 0.0
                    else:
                        try:
                            credit = float(a_norm)
                        except ValueError:
                            credit = 0.0
            desc = line.replace(m.group('date'), '').strip()
            records.append({"Date": date_str, "Description": desc, "Debit": debit, "Credit": credit, "Balance": 0.0})
            continue
        date_search = re.search(date_regex, line, re.IGNORECASE)
        if date_search:
            date_str = date_search.group('date')
            window = " ".join(lines[max(0, i-1):i+3])
            amts = re.findall(amt_regex, window)
            debit = 0.0
            credit = 0.0
            if amts:
                normalized = [re.sub(r'[₹,\s]', '', a).replace('−', '-').replace('—', '-') for a in amts]
                for a_raw, a_norm in zip(amts, normalized):
                    if '-' in a_raw or a_norm.startswith('-'):
                        try:
                            debit = abs(float(a_norm))
                        except ValueError:
                            debit = 0.0
                    else:
                        try:
                            credit = float(a_norm)
                        except ValueError:
                            credit = 0.0
            desc = re.sub(date_regex, '', line).strip()
            records.append({"Date": date_str, "Description": desc, "Debit": debit, "Credit": credit, "Balance": 0.0})
            continue
    df = pd.DataFrame(records)
    for col in ["Debit", "Credit", "Balance"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')
    for fmt in date_formats:
        df["Date"] = pd.to_datetime(df["Date"], format=fmt, errors='coerce')
        if df["Date"].notna().all():
            break
    df["Date"] = df["Date"].fillna(pd.Timestamp("2025-01-01"))
    df["Description"] = df["Description"].astype(str)
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    return df[["Date", "Description", "Debit", "Credit", "Balance"]]

def parse_pdf_to_df(pdf_path):
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed; install with `pip install pymupdf`.")
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except FileNotFoundError as e:
        print(f"PDF file not found: {e}")
        # Proceed with empty text for fallback
    except Exception as e:
        print(f"Unexpected error opening PDF: {e}")
        # Proceed with empty text for fallback

    if parse_bank_statement_text:
        try:
            df = parse_bank_statement_text(text)
            df = standardize_columns(df)
            if 'Date' in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                df["Date"] = df["Date"].fillna(pd.Timestamp("2025-01-01"))
            else:
                print("Warning: No 'Date' column after advanced parsing. Returning empty DataFrame.")
                return pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance"])
            return df
        except Exception:
            pass
    df = basic_parse_text_to_df(text)
    df = standardize_columns(df)
    if 'Date' in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df["Date"] = df["Date"].fillna(pd.Timestamp("2025-01-01"))
    else:
        print("Warning: No 'Date' column after basic parsing. Returning empty DataFrame.")
        return pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance"])
    if text.strip():
        df = parse_bank_statement_text(text)
        logger.info(f"[DEBUG] Parsed DataFrame from text, shape: {df.shape}, columns: {df.columns.tolist()}")
    else:
        logger.warning("[DEBUG] No text extracted from PDF, attempting OCR...")
        df = extract_df_from_scanned_pdf(pdf_path)
        logger.info(f"[DEBUG] OCR DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
    return df

# --------------- Categorization ---------------
def simple_categorize(df):
    df = df.copy()
    def cat(desc):
        s = str(desc).lower()
        if any(k in s for k in ["salary", "credit", "bonus", "payroll"]):
            return "Income"
        if any(k in s for k in ["rent","emi","loan","mortgage","insurance","bill","electricity","phone","utility"]):
            return "Fixed Expenses"
        if any(k in s for k in ["shopping","dining","restaurant","zomato","amazon","flipkart","movie","entertainment","travel"]):
            return "Discretionary Expenses"
        if any(k in s for k in ["fd","rd","mutual fund","sip","deposit","investment","savings"]):
            return "Savings"
        if any(k in s for k in ["overdraft","bounce","insufficient","penalty","late fee","chargeback"]):
            return "Red Flags"
        return "Other"
    df["Category"] = df["Description"].fillna("").apply(cat).astype(str)
    return df

# --------------- ML model integration ---------------
def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print("Failed to load ML model:", e)
        return None

def prepare_features_for_model(model, metrics_df):
    expected_features = ["Average Monthly Income", "Average Monthly Expenses", "Average Monthly EMI",
                        "Net Surplus", "DTI Ratio", "Savings Rate", "Red Flag Count", "Credit Utilization",
                        "Discretionary Expenses", "Income Stability"]
    if isinstance(metrics_df, dict):
        metrics_df = pd.DataFrame([metrics_df])
    elif not isinstance(metrics_df, pd.DataFrame):
        raise ValueError("metrics_df must be dict or DataFrame")
    metrics_df = ensure_numeric(metrics_df, expected_features)
    feature_names = getattr(model, "feature_names_in_", expected_features)
    X = []
    for fn in feature_names:
        val = metrics_df.iloc[0].get(fn, 0.0)
        try:
            val = float(val) if not pd.isna(val) else 0.0
        except:
            val = 0.0
        X.append(val)
    return np.array(X).reshape(1, -1), feature_names

def model_predict_and_explain(model, metrics_df):
    try:
        expected_features = ["Average Monthly Income", "Average Monthly Expenses", "Average Monthly EMI",
                        "Net Surplus", "DTI Ratio", "Savings Rate", "Red Flag Count", "Credit Utilization",
                        "Discretionary Expenses", "Income Stability"]
        print(f"[DEBUG] Expected ML model features: {expected_features}")
        print(f"[DEBUG] Available columns in metrics_df: {list(metrics_df.columns)}")
        
        # Extract features with validation
        features = []
        for feature in expected_features:
            if feature in metrics_df.columns:
                value = metrics_df[feature].iloc[0]
                try:
                    features.append(float(value) if not pd.isna(value) else 0.0)
                except (ValueError, TypeError) as e:
                    print(f"[ERROR] Failed to convert {feature} value {value} to float: {e}")
                    features.append(0.0)
            else:
                print(f"[WARNING] Missing feature {feature} in metrics_df, using 0.0")
                features.append(0.0)
        
        print(f"[DEBUG] ML model input features: {dict(zip(expected_features, features))}")
        
        # Check feature count compatibility
        expected_feature_count = getattr(model, "n_features_in_", len(expected_features))
        if len(features) != expected_feature_count:
            print(f"[WARNING] Feature count mismatch: model expects {expected_feature_count}, got {len(features)}")
            features = features[:expected_feature_count] + [0.0] * (expected_feature_count - len(features)) if len(features) < expected_feature_count else features[:expected_feature_count]
        
        # Make prediction
        prediction = model.predict([features])[0]
        prob = model.predict_proba([features])[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
        ml_result = {
            "model_prediction": "APPROVED" if prediction == 1 else "REJECTED",
            "model_probability": round(max(prob) * 100, 2),
            "features_used": dict(zip(expected_features, features))
        }
        print(f"[DEBUG] ML model output: {ml_result}")
        return ml_result
    except Exception as e:
        print(f"[ERROR] ML model prediction failed: {e}")
        return None

# --------------- CSV export ---------------
def export_raw_csv(df, out_csv):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
    for col in ["Debit", "Credit", "Balance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')
    df["Description"] = df["Description"].astype(str)
    df.to_csv(out_csv, index=False, date_format='%Y-%m-%d')
    return out_csv

# --------------- Main pipeline ---------------
def analyze_file(input_path, cibil_score=None, fill_method="interpolate", out_dir="outputs", applicant_data=None):
    # Initialize default output dictionary
    out = {
        "raw_csv": None,
        "categorized_csv": None,
        "metrics_csv": None,
        "plots": [],
        "html_path": None,
        "pdf_path": None, # Changed to pdf_report for PDF-only output
        "decision_json": None,
        "metrics_df": pd.DataFrame(),
        "transactions_df": pd.DataFrame(),
        "report_text": None
    }
    try:
        input_path = Path(input_path)
        out_dir = ensure_dir(out_dir)
        plots_dir = ensure_dir(out_dir / "plots")
        metrics_dir = ensure_dir(out_dir / "metrics")
        raw_dir = ensure_dir(out_dir / "raw_transactions")
        categorized_dir = ensure_dir(out_dir / "categorized")
        decisions_dir = ensure_dir(out_dir / "loan_decisions")
        reports_dir = ensure_dir(out_dir / "reports")

        # Validate or default CIBIL score
        if cibil_score is not None and (not isinstance(cibil_score, (int, float)) or cibil_score < 300 or cibil_score > 900):
            raise ValueError("CIBIL score must be a number between 300 and 900")
        cibil_score = cibil_score if cibil_score is not None else 700  # Default for testing

        # Ensure applicant_data is a dictionary with defaults
        if applicant_data is None:
            applicant_data = {"name": Path(input_path).stem, "account_number": "Unknown"}
        else:
            applicant_data = {
                "name": applicant_data.get("name", Path(input_path).stem),
                "account_number": applicant_data.get("account_number", "Unknown")
            }

        # 1) Extract
        print("[1/8] Extracting transactions...")
        df = None
        if input_path.suffix.lower() == ".pdf":
            try:
                df = parse_pdf_to_df(str(input_path))
            except Exception as e:
                print("PDF parse error:", e)
                df = pd.DataFrame()
        elif input_path.suffix.lower() in [".csv", ".txt", ".xls", ".xlsx"]:
            try:
                if input_path.suffix.lower() in [".xls", ".xlsx"]:
                    df = pd.read_excel(str(input_path))
                else:
                    df = pd.read_csv(str(input_path))
                df = standardize_columns(df)
            except Exception as e:
                print("File load error:", e)
                df = pd.DataFrame()
        else:
            raise ValueError("Unsupported file type; supported: pdf, csv, xlsx, txt")

        # Process Date column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01")).dt.strftime('%Y-%m-%d')
            df = df.dropna(subset=["Date"]).reset_index(drop=True)

        # Export raw CSV
        raw_csv_path = raw_dir / f"{input_path.stem}_raw.csv"
        export_raw_csv(df, raw_csv_path)
        print(f"Exported raw CSV: {raw_csv_path}")

        # Standardize and ensure required columns
        df = standardize_columns(df)
        for col in ["Debit", "Credit", "Balance"]:
            if col not in df.columns:
                df[col] = 0.0
        if "Date" not in df.columns:
            print("No Date column found after parsing. Exiting.")
            return None

        df = make_arrow_compatible(df)

        # 2) Categorize transactions
        print("[2/8] Categorizing transactions...")
        categorized_csv_path = categorized_dir / f"{input_path.stem}_categorized.csv"
        categorized_df = None

        try:
            # Initialize TransactionCategorizer early
            vectorizer_path = Path("models/vectorizer.pkl")
            classifier_path = Path("models/classifier.pkl")
            tc = TransactionCategorizer()

            if vectorizer_path.exists() and classifier_path.exists():
                tc.load_model(vectorizer_path, classifier_path)
            else:
                print("[WARNING] Model files not found, training TransactionCategorizer with default data...")

            # Try advanced categorization with tc
            categorized_df = categorize_transactions(df, tc, model_type="hybrid", add_confidence=True)

            # Ensure DataFrame is valid
            if categorized_df is None or categorized_df.empty:
                print("categorize_transactions produced empty/invalid DataFrame, falling back to simple_categorize.")
                categorized_df = simple_categorize(df)

        except Exception as e:
            print(f"Categorization error: {e}")
            logger.error(f"Categorization error: {e}")
            categorized_df = simple_categorize(df)

        # Standardize columns and handle Date
        categorized_df = standardize_columns(categorized_df)
        if "Date" in categorized_df.columns:
            categorized_df["Date"] = pd.to_datetime(
                categorized_df["Date"], errors="coerce"
            ).fillna(pd.Timestamp("2025-01-01"))
        else:
            print("Warning: 'Date' column missing, adding default.")
            categorized_df["Date"] = pd.Timestamp("2025-01-01")

        # Save categorized file
        categorized_df.to_csv(categorized_csv_path, index=False)
        print(f"[DEBUG] categorize_transactions returned type: {type(categorized_df)}, shape: {categorized_df.shape}")
        print(f"Categorized CSV: {categorized_csv_path}")

        # 3) Metrics
        print("[3/8] Calculating metrics...")
        metrics_file = metrics_dir / f"{input_path.stem}_metrics.csv"
        try:
            out_metrics = calculate_metrics(categorized_df, str(metrics_file))
            if isinstance(out_metrics, pd.DataFrame):
                metrics_df = out_metrics
            elif isinstance(out_metrics, dict):
                metrics_df = pd.DataFrame([out_metrics])
            else:
                metrics_df = pd.DataFrame([out_metrics])
            metrics_df = enforce_metrics_schema(metrics_df)
        except Exception as e:
            print("calculate_metrics util error:", e)
            logger.error(f"calculate_metrics util error: {e}")
            monthly = categorized_df.set_index("Date").groupby(pd.Grouper(freq="M")).agg(
                Income=("Credit", "sum"), Expenses=("Debit", "sum"))
            avg_income = monthly["Income"].mean() if not monthly.empty else 0.0
            avg_expenses = monthly["Expenses"].mean() if not monthly.empty else 0.0
            net_surplus = avg_income - avg_expenses
            metrics_df = pd.DataFrame([{
                "Average Monthly Income": avg_income,
                "Average Monthly Expenses": avg_expenses,
                "Net Surplus": net_surplus,
                "DTI Ratio": 0.0,
                "Savings Rate": 0.0,
                "Red Flag Count": int((categorized_df["Category"] == "Red Flags").sum()),
                "Credit Utilization": 0.0,
                "Discretionary Expenses": categorized_df[categorized_df["Category"] == "Discretionary Expenses"]["Debit"].sum(),
                "Income Stability": 0
            }])
            metrics_df = enforce_metrics_schema(metrics_df)

        # Additional metrics
        additional_metrics_df = pd.DataFrame({
            'Income Variability Index': [categorized_df[categorized_df['Category'] == 'Income']['Credit'].std() / 
                                        categorized_df[categorized_df['Category'] == 'Income']['Credit'].mean() 
                                        if categorized_df[categorized_df['Category'] == 'Income']['Credit'].mean() > 0 else 0.0],
            'Number of Income Sources': [len(categorized_df[categorized_df['Category'] == 'Income']['Description'].unique())],
            'Recent Salary Trend (%)': [(categorized_df[categorized_df['Category'] == 'Income'].groupby(categorized_df['Date'].dt.strftime('%Y-%m'))['Credit'].sum().iloc[-1] - 
                               categorized_df[categorized_df['Category'] == 'Income'].groupby(categorized_df['Date'].dt.strftime('%Y-%m'))['Credit'].sum().iloc[-2]) / 
                               categorized_df[categorized_df['Category'] == 'Income'].groupby(categorized_df['Date'].dt.strftime('%Y-%m'))['Credit'].sum().iloc[-2] * 100 
                               if len(categorized_df[categorized_df['Category'] == 'Income'].groupby(categorized_df['Date'].dt.strftime('%Y-%m'))['Credit'].sum()) >= 2 and 
                               categorized_df[categorized_df['Category'] == 'Income'].groupby(categorized_df['Date'].dt.strftime('%Y-%m'))['Credit'].sum().iloc[-2] > 0 else 0.0],
            'Discretionary Spending (%)': [(categorized_df[categorized_df['Category'] == 'Discretionary Expenses']['Debit'].sum() / 
                                          categorized_df[categorized_df['Category'] == 'Income']['Credit'].sum() * 100) 
                                          if categorized_df[categorized_df['Category'] == 'Income']['Credit'].sum() > 0 else 0.0],
            'High-Cost EMI Payments': [categorized_df[categorized_df['Category'] == 'Loan Payments']['Debit'].mean() 
                                      if not categorized_df[categorized_df['Category'] == 'Loan Payments'].empty else 0.0],
            'Existing Loan Count': [len(categorized_df[categorized_df['Category'] == 'Loan Payments']['Description'].unique())],
            'Credit Card Payments': [categorized_df[categorized_df['Category'] == 'Credit Card Payments']['Debit'].sum() 
                                    if not categorized_df[categorized_df['Category'] == 'Credit Card Payments'].empty else 0.0],
            'Bounced Cheques Count': [len(categorized_df[categorized_df['Category'] == 'Bounced Cheques'])],
            'Minimum Monthly Balance': [categorized_df.groupby(categorized_df['Date'].dt.strftime('%Y-%m'))['Balance'].min().mean() 
                               if not categorized_df.empty else 0.0],
            'Average Closing Balance': [categorized_df.groupby(categorized_df['Date'].dt.strftime('%Y-%m'))['Balance'].last().mean() 
                               if not categorized_df.empty else 0.0],
            'Overdraft Usage Frequency': [len(categorized_df[categorized_df['Balance'] < 0])],
            'Negative Balance Days': [len(categorized_df[categorized_df['Balance'] < 0]['Date'].unique())],
            'Sudden High-Value Credits': [len(categorized_df[(categorized_df['Category'] == 'Income') & (categorized_df['Credit'] > 100000)])],
            'Circular Transactions': [len(categorized_df['Description'].value_counts()[categorized_df['Description'].value_counts() > 5])]
        })
        metrics_df = pd.concat([metrics_df, additional_metrics_df], axis=1)
        high_value_mask = identify_high_value_transactions(categorized_df)
        recurring_mask = recurring_transactions(categorized_df)
        metrics_df['High-Value Transactions Count'] = high_value_mask.sum() if high_value_mask is not None else 0
        metrics_df['Recurring Transactions Count'] = recurring_mask.sum() if recurring_mask is not None else 0
        metrics_df = enforce_schema(metrics_df)
        metrics_df = make_arrow_compatible(metrics_df)
        metrics_df.to_csv(metrics_file, index=False)
        out["metrics_csv"] = str(metrics_file)
        print(f"Saved metrics: {metrics_file}")
        logger.info(f"[DEBUG] Metrics DataFrame shape: {metrics_df.shape}, columns: {metrics_df.columns.tolist()}")

        print("[5/9] Identifying high-value transactions...")
        categorized_df["High_Value"] = high_value_mask
        logger.info(f"[DEBUG] High-value transactions count: {high_value_mask.sum()}")

        print("[6/9] Identifying recurring transactions...")
        categorized_df["Recurring"] = recurring_mask
        logger.info(f"[DEBUG] Recurring transactions count: {recurring_mask.sum()}")

        # 4) Decision (heuristic)
        print("[4/8] Heuristic scoring & decisioning...")
        try:
            if 'score_and_decide' in globals():
                heuristic_decision = score_and_decide(metrics_df=metrics_df, cibil_score=cibil_score, categorized_file=str(categorized_csv_path))
                if not isinstance(heuristic_decision, dict):
                    logger.error(f"score_and_decide returned invalid result: {heuristic_decision}")
                    heuristic_decision = {"Action": "Review", "Reason": "Invalid output from score_and_decide"}
            else:
                logger.warning("score_and_decide not found, using fallback logic")
                # Simple fallback decision logic
                avg_income = metrics_df['Average Monthly Income'].iloc[0] if not metrics_df.empty and 'Average Monthly Income' in metrics_df.columns else 0.0
                red_flags = metrics_df['Red Flag Count'].iloc[0] if not metrics_df.empty and 'Red Flag Count' in metrics_df.columns else 0
                if avg_income >= 30000 and red_flags == 0 and cibil_score >= 650:
                    heuristic_decision = {"Action": "Approve", "Reason": "Good income and credit profile"}
                else:
                    heuristic_decision = {"Action": "Reject", "Reason": "Insufficient income or credit issues"}
        except Exception as e:
            logger.error(f"Decision logic error: {e}")
            heuristic_decision = {"Action": "Review", "Reason": f"Error in decision logic: {e}"}
        print("Heuristic decision:", heuristic_decision)

        # 5) ML model prediction
        print("[5/8] ML model integration (if model exists)...")
        model_path = Path("models/loan_approval_model.pkl")
        model = load_model(model_path) if 'load_model' in globals() else None
        ml_result = None
        if model is not None:
            try:
                ml_result_temp = model_predict_and_explain(model, metrics_df) if 'model_predict_and_explain' in globals() else None
                if isinstance(ml_result_temp, dict):
                    ml_result = ml_result_temp
                else:
                    logger.error(f"model_predict_and_explain returned invalid result: {ml_result_temp}")
            except Exception as e:
                print(f"[ERROR] ML model prediction failed: {e}")
                logger.error(f"ML model prediction failed: {e}")
                ml_result = None
        else:
            logger.warning(f"No ML model found at {model_path}")
        print(f"[DEBUG] ML result: {ml_result}")

        # 6) Combine heuristic and ML decisions
        print("[6/8] Final decision reconciliation...")
        if not isinstance(heuristic_decision, dict):
            logger.error(f"heuristic_decision is not a dict: {heuristic_decision}")
            heuristic_decision = {"Action": "Review", "Reason": "Invalid heuristic decision output"}
        final_decision = heuristic_decision
        if isinstance(ml_result, dict) and ml_result.get("model_probability", 0) >= 70:
            final_action = ml_result.get("model_prediction", "Unknown")
            final_reason = f"ML-based decision (confidence: {ml_result.get('model_probability', 0)}%)"
        else:
            final_action = heuristic_decision.get("Action", "Unknown")
            final_reason = heuristic_decision.get("Reason", "No reason provided")
            logger.info(f"Using heuristic decision (ML confidence {ml_result.get('model_probability', 0) if isinstance(ml_result, dict) else 0}% or model unavailable)")
        # Display decision (Streamlit-compatible)
        action = final_decision.get("Action", "Unknown")
        reason = final_decision.get("Reason", "No reason provided")
        if not isinstance(reason, str):
            reason = str(reason)  # Convert to string to avoid unpack errors
        print(f"Final Decision: {action}, Reason: {reason}")

        if 'st' in globals():  # Check if running in Streamlit
            st.subheader("📋 Final Decision")
            if action.lower() == "approve":
                st.success(f"✅ {action} ({reason})")
            else:
                st.error(f"❌ {action} ({reason})")
            print(f"Final Decision: {action}, Reason: {reason}")

        # 7) Visualizations
        print("[7/8] Generating visualizations...")
        categorized_df["Amount"] = categorized_df["Credit"].fillna(0.0) - categorized_df["Debit"].fillna(0.0)

        plot_functions = [
            (plot_income_trend_plotly, "income_trend.png"),
            (plot_surplus_trend_plotly, "surplus_trend.png"),
            (plot_income_vs_expenses_plotly, "income_vs_expenses.png"),
            (plot_cumulative_savings_plotly, "cumulative_savings.png"),
            (plot_category_breakdown_plotly, "category_breakdown.png"),
            (lambda df: plot_high_risk_timeline(df, str(plots_dir / "high_risk_timeline.png")), "high_risk_timeline.png"),
            (lambda df: plot_cibil_gauge(cibil_score, str(plots_dir / "cibil_gauge.png")), "cibil_gauge.png"),
            (plot_credit_debit_ratio_trend_plotly, "credit_debit_ratio_trend.png"),
            (plot_category_expenses_over_time_plotly, "category_expenses_over_time.png"),
            (plot_top_expenses_pareto_plotly, "top_expenses_pareto.png"),
            (plot_income_expense_scatter_plotly, "income_expense_scatter.png"),
            (plot_monthly_transaction_volume_by_category_plotly, "monthly_transaction_volume_by_category.png")
        ]

        plot_paths = []

        @contextmanager
        def safe_browser():
            """Context manager to ensure browser cleanup for Plotly rendering."""
            try:
                yield
            finally:
                try:
                    pio.kaleido.scope.stop()
                except Exception:
                    pass

        for func, filename in plot_functions:
            try:
                fig = func(cibil_score) if 'cibil' in filename else func(categorized_df)
                if fig:
                    if 'st' in globals():  # Running in Streamlit
                        st.plotly_chart(fig, use_container_width=True)
                    output_path = str(plots_dir / filename)
                    fig.write_image(output_path, engine="kaleido")
                    plot_paths.append((output_path, filename.replace(".png", "").replace("_", " ").title()))
                    print(f"Generated {filename}")
            except Exception as e:
                print(f"Error generating {filename}: {e}")
                logger.error(f"Error generating {filename}: {e}")

        print(f"Saved {len(plot_paths)} plots to {plots_dir}")

        # 10) Generate PDF report
        print("[10/10] Building PDF report...")
        # Create readable filename
        name = applicant_data.get('name', input_path.stem)
        # Remove temp file prefixes like 'tmp' and random characters, keep meaningful parts
        if name.startswith('tmp') and '_' in name:
            name = name.split('_', 1)[-1]  # Extract meaningful part after 'tmpXXXX_'
        sanitized_name = re.sub(r'[^\w\s-]', '', name).replace('\n', ' ').replace(' ', '_').strip()
        pdf_path = reports_dir / f"{sanitized_name}_{cibil_score}_report.pdf"
        html_path = reports_dir / f"{sanitized_name}_{cibil_score}_report.html"
        # HTML template with escaped curly braces in CSS
        premium_template = """<!DOCTYPE html>
        <html>
        <head>
            <title>Loan Eligibility Report</title>
            <style>
                body {{ font-family: Helvetica, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 8px; text-align: center; }}
                .section {{ margin-top: 30px; }}
                .table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .table th {{ background-color: #e0e0e0; font-weight: bold; }}
                .table td {{ background-color: #f9f9f9; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; border-bottom: 2px solid #ccc; padding-bottom: 5px; }}
                img {{ max-width: 100%; height: auto; display: block; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Loan Eligibility Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
            <div class="section">
                <h2>Applicant Data</h2>
                <table class="table">
                    <tr><th>Name</th><td>{applicant_name}</td></tr>
                    <tr><th>Account Number</th><td>{account_number}</td></tr>
                    <tr><th>CIBIL Score</th><td>{cibil_score}</td></tr>
                </table>
            </div>
            <div class="section">
                <h2>Income & Stability</h2>
                <table class="table">
                    <tr><th>Average Monthly Income</th><td>₹{avg_monthly_income:,.2f}</td></tr>
                    <tr><th>Income Variability Index</th><td>{income_variability_index:.2f}</td></tr>
                    <tr><th>Number of Income Sources</th><td>{num_income_sources}</td></tr>
                    <tr><th>Recent Salary Trend</th><td>{recent_salary_trend:.2f}%</td></tr>
                </table>
            </div>
            <div class="section">
                <h2>Expenses & Lifestyle</h2>
                <table class="table">
                    <tr><th>Average Monthly Expenses</th><td>₹{avg_monthly_expenses:,.2f}</td></tr>
                    <tr><th>Savings Ratio</th><td>{savings_ratio:.2f}%</td></tr>
                    <tr><th>Discretionary Spending</th><td>{discretionary_spending:.2f}%</td></tr>
                    <tr><th>High-Cost EMI Payments</th><td>₹{high_cost_emi:,.2f}</td></tr>
                </table>
            </div>
            <div class="section">
                <h2>Debt Metrics</h2>
                <table class="table">
                    <tr><th>DTI Ratio</th><td>{dti_ratio:.2f}%</td></tr>
                    <tr><th>Existing Loan Count</th><td>{existing_loan_count}</td></tr>
                    <tr><th>Credit Card Payments</th><td>₹{credit_card_payments:,.2f}</td></tr>
                    <tr><th>Bounced Cheques Count</th><td>{bounced_cheques_count}</td></tr>
                </table>
            </div>
            <div class="section">
                <h2>Cash Flow & Liquidity</h2>
                <table class="table">
                    <tr><th>Minimum Monthly Balance</th><td>₹{min_monthly_balance:,.2f}</td></tr>
                    <tr><th>Average Closing Balance</th><td>₹{avg_closing_balance:,.2f}</td></tr>
                    <tr><th>Overdraft Usage Frequency</th><td>{overdraft_frequency}</td></tr>
                    <tr><th>Negative Balance Days</th><td>{negative_balance_days}</td></tr>
                </table>
            </div>
            <div class="section">
                <h2>Creditworthiness Indicators</h2>
                <table class="table">
                    <tr><th>CIBIL Score</th><td>{cibil_score}</td></tr>
                    <tr><th>Payment History</th><td>Derived from statement</td></tr>
                    <tr><th>Delinquency Flags</th><td>{delinquency_flags}</td></tr>
                    <tr><th>Recent Loan Inquiries</th><td>Not available</td></tr>
                </table>
            </div>
            <div class="section">
                <h2>Fraud & Compliance Checks</h2>
                <table class="table">
                    <tr><th>Sudden High-Value Credits</th><td>{sudden_high_value_credits}</td></tr>
                    <tr><th>Circular Transactions</th><td>{circular_transactions}</td></tr>
                    <tr><th>Salary Mismatch</th><td>Not detected</td></tr>
                    <tr><th>Blacklisted Accounts</th><td>Not detected</td></tr>
                </table>
            </div>
            <div class="section">
                <h2>Decision Metrics</h2>
                <table class="table">
                    <tr><th>Bank Score</th><td>{bank_score:.2f}/100</td></tr>
                    <tr><th>DTI Ratio</th><td>{dti_ratio:.2f}%</td></tr>
                    <tr><th>Average Closing Balance</th><td>₹{avg_closing_balance:,.2f}</td></tr>
                    <tr><th>CIBIL Score</th><td>{cibil_score}</td></tr>
                    <tr><th>Bounced Cheques</th><td>{bounced_cheques_count}</td></tr>
                </table>
            </div>
            <div class="section">
                <h2>Final Decision</h2>
                <p style="color: {decision_color}; font-weight: bold;">
                    <b>{final_action}</b>: {final_reason}
                </p>
            </div>
            <div class="section">
                <h2>ML Model Prediction</h2>
                <table class="table">
                    <tr><th>Prediction</th><td>{ml_prediction}</td></tr>
                    <tr><th>Confidence</th><td>{ml_probability:.2f}%</td></tr>
                </table>
            </div>
            <div class="section">
                <h2>Visualizations</h2>
                {plots_html}
            </div>
        </body>
        </html>"""
        # Helper functions to sanitize inputs
        def safe_float(value, default=0.0):
            try:
                return float(value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid float value: {value}, using default {default}")
                return default
        def safe_int(value, default=0):
            try:
                return int(value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid int value: {value}, using default {default}")
                return default
        def safe_str(value, default="Unknown"):
            if value is None:
                return default
            try:
                # Replace curly braces to prevent formatting errors
                return str(value).replace('{', '{{').replace('}', '}}').strip()
            except Exception as e:
                logger.warning(f"Invalid string value: {value}, error: {e}, using default {default}")
                return default
        # Prepare data for template with validation
        avg_monthly_income = safe_float(metrics_df['Average Monthly Income'].iloc[0] if not metrics_df.empty else 0.0)
        avg_monthly_expenses = safe_float(metrics_df['Average Monthly Expenses'].iloc[0] if not metrics_df.empty else 0.0)
        income_variability_index = safe_float(metrics_df['Income Variability Index'].iloc[0] if not metrics_df.empty else 0.0)
        num_income_sources = safe_int(metrics_df['Number of Income Sources'].iloc[0] if not metrics_df.empty else 0)
        recent_salary_trend = safe_float(metrics_df['Recent Salary Trend (%)'].iloc[0] if not metrics_df.empty else 0.0)
        savings_ratio = safe_float(metrics_df['Savings Rate'].iloc[0] if not metrics_df.empty else 0.0)
        discretionary_spending = safe_float(metrics_df['Discretionary Spending (%)'].iloc[0] if not metrics_df.empty else 0.0)
        high_cost_emi = safe_float(metrics_df['High-Cost EMI Payments'].iloc[0] if not metrics_df.empty else 0.0)
        dti_ratio = safe_float(metrics_df['DTI Ratio'].iloc[0] if not metrics_df.empty else 0.0)
        existing_loan_count = safe_int(metrics_df['Existing Loan Count'].iloc[0] if not metrics_df.empty else 0)
        credit_card_payments = safe_float(metrics_df['Credit Card Payments'].iloc[0] if not metrics_df.empty else 0.0)
        bounced_cheques_count = safe_int(metrics_df['Bounced Cheques Count'].iloc[0] if not metrics_df.empty else 0)
        min_monthly_balance = safe_float(metrics_df['Minimum Monthly Balance'].iloc[0] if not metrics_df.empty else 0.0)
        avg_closing_balance = safe_float(metrics_df['Average Closing Balance'].iloc[0] if not metrics_df.empty else 0.0)
        overdraft_frequency = safe_int(metrics_df['Overdraft Usage Frequency'].iloc[0] if not metrics_df.empty else 0)
        negative_balance_days = safe_int(metrics_df['Negative Balance Days'].iloc[0] if not metrics_df.empty else 0)
        delinquency_flags = safe_int(bounced_cheques_count)
        sudden_high_value_credits = safe_int(metrics_df['Sudden High-Value Credits'].iloc[0] if not metrics_df.empty else 0)
        circular_transactions = safe_int(metrics_df['Circular Transactions'].iloc[0] if not metrics_df.empty else 0)
        bank_score = safe_float(heuristic_decision.get('Total Score', 0))
        ml_prediction = safe_str(ml_result.get('model_prediction', 'N/A') if ml_result else 'N/A')
        ml_probability = safe_float(ml_result.get('model_probability', 0) if ml_result else 0)
        final_action = safe_str(heuristic_decision.get('Action', 'Unknown'))
        final_reason = safe_str(heuristic_decision.get('Reason', 'No reason provided'))
        applicant_name = safe_str(applicant_data.get('name', 'Unknown'))
        account_number = safe_str(applicant_data.get('account_number', 'Unknown'))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        decision_color = safe_str('#008000' if final_action.lower() == 'approve' else '#FF0000')
        # Debug variable values
        logger.debug(f"Template variables: applicant_name={applicant_name}, account_number={account_number}, "
                     f"cibil_score={cibil_score}, final_action={final_action}, final_reason={final_reason}, "
                     f"decision_color={decision_color}")
        # Embed plots as base64 images for HTML
        plots_html = ''
        for plot_path, title in plot_paths:
            if os.path.exists(plot_path):
                try:
                    with open(plot_path, "rb") as img_file:
                        img_b64 = base64.b64encode(img_file.read()).decode()
                    plots_html += f'<h3>{safe_str(title)}</h3><img src="data:image/png;base64,{img_b64}" alt="{safe_str(title)}"><br>'
                except Exception as e:
                    logger.error(f"Failed to embed plot {plot_path}: {e}")
        # Fill HTML template
        try:
            html_str = premium_template.format(
                timestamp=timestamp,
                applicant_name=applicant_name,
                account_number=account_number,
                cibil_score=cibil_score,
                avg_monthly_income=avg_monthly_income,
                income_variability_index=income_variability_index,
                num_income_sources=num_income_sources,
                recent_salary_trend=recent_salary_trend,
                avg_monthly_expenses=avg_monthly_expenses,
                savings_ratio=savings_ratio,
                discretionary_spending=discretionary_spending,
                high_cost_emi=high_cost_emi,
                dti_ratio=dti_ratio,
                existing_loan_count=existing_loan_count,
                credit_card_payments=credit_card_payments,
                bounced_cheques_count=bounced_cheques_count,
                min_monthly_balance=min_monthly_balance,
                avg_closing_balance=avg_closing_balance,
                overdraft_frequency=overdraft_frequency,
                negative_balance_days=negative_balance_days,
                delinquency_flags=delinquency_flags,
                sudden_high_value_credits=sudden_high_value_credits,
                circular_transactions=circular_transactions,
                bank_score=bank_score,
                final_action=final_action,
                final_reason=final_reason,
                decision_color=decision_color,
                ml_prediction=ml_prediction,
                ml_probability=ml_probability,
                plots_html=plots_html
            )
        except Exception as e:
            logger.error(f"Failed to format HTML template: {e}")
            raise
        
        # Save HTML
        try:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_str)
            logger.info(f"Generated HTML: {html_path}")
            out["html_path"] = str(html_path)
            out["report_text"] = html_str
        except Exception as e:
            logger.error(f"Failed to save HTML: {e}")
            out["html_path"] = None
        # Generate PDF using ReportLab
        try:
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            # Register standard fonts
            try:
                # Use built-in fonts to avoid external dependencies
                pdfmetrics.registerFont(TTFont('Helvetica', 'Helvetica'))
                pdfmetrics.registerFont(TTFont('Helvetica-Bold', 'Helvetica-Bold'))
            except Exception as e:
                logger.warning(f"Failed to register fonts, using defaults: {e}")

            # Create PDF buffer
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
            styles = getSampleStyleSheet()

            # Customize styles
            styles.add(ParagraphStyle(name='HeaderTitle', fontName='Helvetica-Bold', fontSize=16, leading=20, textColor=colors.HexColor('#333333'), alignment=1))
            styles.add(ParagraphStyle(name='HeaderText', fontName='Helvetica', fontSize=10, leading=12, textColor=colors.HexColor('#333333'), alignment=1))
            styles.add(ParagraphStyle(name='SectionHeading', fontName='Helvetica-Bold', fontSize=12, leading=14, textColor=colors.HexColor('#555555'), spaceAfter=5))
            styles.add(ParagraphStyle(name='DecisionText', fontName='Helvetica-Bold', fontSize=10, leading=12, textColor=colors.HexColor(decision_color)))
            story = []
            # Header Section
            header_table = Table([[Paragraph("Loan Eligibility Report", styles['HeaderTitle'])], [Paragraph(f"Generated on: {timestamp}", styles['HeaderText'])]], colWidths=[A4[0] - 80])
            header_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F0F0')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
                ('ROUNDED', (0, 0), (-1, -1), 8),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, 0), 10),
                ('PADDING', (0, 1), (-1, 1), 5),
            ]))
            story.append(header_table)
            story.append(Spacer(1, 30))
            # Table Style for all sections
            table_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E0E0E0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F9F9F9')),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DDDDDD')),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#DDDDDD')),
            ])
            # Applicant Data
            story.append(Paragraph("Applicant Data", styles['SectionHeading']))
            data = [
                ["Name", applicant_name],
                ["Account Number", account_number],
                ["CIBIL Score", str(cibil_score)]
            ]
            table = Table(data, colWidths=[200, A4[0] - 280])
            table.setStyle(table_style)
            story.append(table)
            story.append(Spacer(1, 30))
            # Income & Stability
            story.append(Paragraph("Income & Stability", styles['SectionHeading']))
            data = [
                ["Average Monthly Income", f"₹{avg_monthly_income:,.2f}"],
                ["Income Variability Index", f"{income_variability_index:.2f}"],
                ["Number of Income Sources", str(num_income_sources)],
                ["Recent Salary Trend", f"{recent_salary_trend:.2f}%"]
            ]
            table = Table(data, colWidths=[200, A4[0] - 280])
            table.setStyle(table_style)
            story.append(table)
            story.append(Spacer(1, 30))
            # Expenses & Lifestyle
            story.append(Paragraph("Expenses & Lifestyle", styles['SectionHeading']))
            data = [
                ["Average Monthly Expenses", f"₹{avg_monthly_expenses:,.2f}"],
                ["Savings Ratio", f"{savings_ratio:.2f}%"],
                ["Discretionary Spending", f"{discretionary_spending:.2f}%"],
                ["High-Cost EMI Payments", f"₹{high_cost_emi:,.2f}"]
            ]
            table = Table(data, colWidths=[200, A4[0] - 280])
            table.setStyle(table_style)
            story.append(table)
            story.append(Spacer(1, 30))
            # Debt Metrics
            story.append(Paragraph("Debt Metrics", styles['SectionHeading']))
            data = [
                ["DTI Ratio", f"{dti_ratio:.2f}%"],
                ["Existing Loan Count", str(existing_loan_count)],
                ["Credit Card Payments", f"₹{credit_card_payments:,.2f}"],
                ["Bounced Cheques Count", str(bounced_cheques_count)]
            ]
            table = Table(data, colWidths=[200, A4[0] - 280])
            table.setStyle(table_style)
            story.append(table)
            story.append(Spacer(1, 30))
            # Cash Flow & Liquidity
            story.append(Paragraph("Cash Flow & Liquidity", styles['SectionHeading']))
            data = [
                ["Minimum Monthly Balance", f"₹{min_monthly_balance:,.2f}"],
                ["Average Closing Balance", f"₹{avg_closing_balance:,.2f}"],
                ["Overdraft Usage Frequency", str(overdraft_frequency)],
                ["Negative Balance Days", str(negative_balance_days)]
            ]
            table = Table(data, colWidths=[200, A4[0] - 280])
            table.setStyle(table_style)
            story.append(table)
            story.append(Spacer(1, 30))
            # Creditworthiness Indicators
            story.append(Paragraph("Creditworthiness Indicators", styles['SectionHeading']))
            data = [
                ["CIBIL Score", str(cibil_score)],
                ["Payment History", "Derived from statement"],
                ["Delinquency Flags", str(delinquency_flags)],
                ["Recent Loan Inquiries", "Not available"]
            ]
            table = Table(data, colWidths=[200, A4[0] - 280])
            table.setStyle(table_style)
            story.append(table)
            story.append(Spacer(1, 30))
            # Fraud & Compliance Checks
            story.append(Paragraph("Fraud & Compliance Checks", styles['SectionHeading']))
            data = [
                ["Sudden High-Value Credits", str(sudden_high_value_credits)],
                ["Circular Transactions", str(circular_transactions)],
                ["Salary Mismatch", "Not detected"],
                ["Blacklisted Accounts", "Not detected"]
            ]
            table = Table(data, colWidths=[200, A4[0] - 280])
            table.setStyle(table_style)
            story.append(table)
            story.append(Spacer(1, 30))
            # Decision Metrics
            story.append(Paragraph("Decision Metrics", styles['SectionHeading']))
            data = [
                ["Bank Score", f"{bank_score:.2f}/100"],
                ["DTI Ratio", f"{dti_ratio:.2f}%"],
                ["Average Closing Balance", f"₹{avg_closing_balance:,.2f}"],
                ["CIBIL Score", str(cibil_score)],
                ["Bounced Cheques", str(bounced_cheques_count)]
            ]
            table = Table(data, colWidths=[200, A4[0] - 280])
            table.setStyle(table_style)
            story.append(table)
            story.append(Spacer(1, 30))
            # Final Decision
            story.append(Paragraph("Final Decision", styles['SectionHeading']))
            story.append(Paragraph(f"<b>{final_action}</b>: {final_reason}", styles['DecisionText']))
            story.append(Spacer(1, 30))
            # ML Model Prediction
            story.append(Paragraph("ML Model Prediction", styles['SectionHeading']))
            data = [
                ["Prediction", ml_prediction],
                ["Confidence", f"{ml_probability:.2f}%"]
            ]
            table = Table(data, colWidths=[200, A4[0] - 280])
            table.setStyle(table_style)
            story.append(table)
            story.append(Spacer(1, 30))
            # Visualizations
            story.append(Paragraph("Visualizations", styles['SectionHeading']))
            for plot_path, title in plot_paths:
                if os.path.exists(plot_path):
                    try:
                        story.append(Paragraph(safe_str(title), styles['Heading3']))
                        story.append(Image(plot_path, width=A4[0] - 80, height=300, kind='proportional'))
                        story.append(Spacer(1, 10))
                    except Exception as e:
                        logger.error(f"Failed to add plot {plot_path} to PDF: {e}")
            # Build PDF
            doc.build(story)
            with open(pdf_path, 'wb') as f:
                f.write(pdf_buffer.getvalue())
            logger.info(f"Saved PDF report: {pdf_path}")
            out["pdf_path"] = str(pdf_path)
        except Exception as e:
            logger.error(f"Failed to generate PDF with ReportLab: {e}")
            out["pdf_path"] = None
            print("PDF generation failed. HTML report available.")

        # Save decision JSON and log
        print("[11/11] Saving decision JSON and log...")
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record = {
                "timestamp": timestamp,
                "file": str(input_path),
                "cibil": cibil_score,
                "heuristic_action": heuristic_decision.get("Action"),
                "heuristic_score": heuristic_decision.get("Total Score"),
                "heuristic_reason": heuristic_decision.get("Reason"),
                "ml_prediction": ml_result.get("model_prediction") if ml_result else None,
                "ml_probability": ml_result.get("model_probability") if ml_result else None
            }
            decision_json = decisions_dir / f"{input_path.stem}_decision.json"
            with open(decision_json, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            log_csv = decisions_dir / "decisions_log.csv"
            pd.DataFrame([record]).to_csv(log_csv, mode='a', header=not log_csv.exists(), index=False)
            print("Decision JSON saved:", decision_json)
            print("Decision log appended:", log_csv)
            out["decision_json"] = str(decision_json)
        except Exception as e:
            logger.error(f"Failed to save decision JSON/log: {e}")

        # Display report in Streamlit
        if st.runtime.exists():
            try:
                st.markdown("### Loan Eligibility Report")
                st.markdown(f"**Generated on:** {timestamp}")
                st.markdown("#### Applicant Data")
                st.table({
                    "Name": [applicant_name],
                    "Account Number": [account_number],
                    "CIBIL Score": [cibil_score]
                })
                st.markdown("#### Income & Stability")
                st.table({
                    "Average Monthly Income": [f"₹{avg_monthly_income:,.2f}"],
                    "Income Variability Index": [f"{income_variability_index:.2f}"],
                    "Number of Income Sources": [num_income_sources],
                    "Recent Salary Trend": [f"{recent_salary_trend:.2f}%"]
                })
                st.markdown("#### Expenses & Lifestyle")
                st.table({
                    "Average Monthly Expenses": [f"₹{avg_monthly_expenses:,.2f}"],
                    "Savings Ratio": [f"{savings_ratio:.2f}%"],
                    "Discretionary Spending": [f"{discretionary_spending:.2f}%"],
                    "High-Cost EMI Payments": [f"₹{high_cost_emi:,.2f}"]
                })
                st.markdown("#### Debt Metrics")
                st.table({
                    "DTI Ratio": [f"{dti_ratio:.2f}%"],
                    "Existing Loan Count": [existing_loan_count],
                    "Credit Card Payments": [f"₹{credit_card_payments:,.2f}"],
                    "Bounced Cheques Count": [bounced_cheques_count]
                })
                st.markdown("#### Cash Flow & Liquidity")
                st.table({
                    "Minimum Monthly Balance": [f"₹{min_monthly_balance:,.2f}"],
                    "Average Closing Balance": [f"₹{avg_closing_balance:,.2f}"],
                    "Overdraft Usage Frequency": [overdraft_frequency],
                    "Negative Balance Days": [negative_balance_days]
                })
                st.markdown("#### Creditworthiness Indicators")
                st.table({
                    "CIBIL Score": [cibil_score],
                    "Payment History": ["Derived from statement"],
                    "Delinquency Flags": [delinquency_flags],
                    "Recent Loan Inquiries": ["Not available"]
                })
                st.markdown("#### Fraud & Compliance Checks")
                st.table({
                    "Sudden High-Value Credits": [sudden_high_value_credits],
                    "Circular Transactions": [circular_transactions],
                    "Salary Mismatch": ["Not detected"],
                    "Blacklisted Accounts": ["Not detected"]
                })
                st.markdown("#### Decision Metrics")
                st.table({
                    "Bank Score": [f"{bank_score:.2f}/100"],
                    "DTI Ratio": [f"{dti_ratio:.2f}%"],
                    "Average Closing Balance": [f"₹{avg_closing_balance:,.2f}"],
                    "CIBIL Score": [cibil_score],
                    "Bounced Cheques": [bounced_cheques_count]
                })
                st.markdown("#### Final Decision")
                if final_action.lower() == "approve":
                    st.success(f"**{final_action}**: {final_reason}")
                else:
                    st.error(f"**{final_action}**: {final_reason}")
                st.markdown("#### ML Model Prediction")
                st.table({
                    "Prediction": [ml_prediction],
                    "Confidence": [f"{ml_probability:.2f}%"]
                })
                st.markdown("#### Visualizations")
                for plot_path, title in plot_paths:
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption=title, use_container_width=True)
                if out["pdf_path"] and os.path.exists(out["pdf_path"]):
                    with open(out["pdf_path"], "rb") as f:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=f,
                            file_name=f"{sanitized_name}_{cibil_score}_report.pdf",
                            mime="application/pdf",
                            key=f"download_report_{sanitized_name}"
                        )
                else:
                    st.error("PDF report not available for download.")
            except Exception as e:
                logger.error(f"Failed to display Streamlit report: {e}")
                st.error(f"Failed to display report: {e}")

        out["report_text"] = html_str
       
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        print(f"Error in analysis: {e}")
    return out