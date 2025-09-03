# visualizer.py
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
import plotly.io as pio
from contextlib import contextmanager

# Setup logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helper functions
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
    df["DTI Ratio"] = np.where(
        df["Average Monthly Income"] > 0,
        (df["Average Monthly EMI"] / df["Average Monthly Income"]) * 100.0,
        0.0
    )
    df["Savings Rate"] = df["Savings Rate"].astype("float64")
    df["DTI Ratio"] = df["DTI Ratio"].astype("float64")
    return df

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
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except FileNotFoundError as e:
        print(f"PDF file not found: {e}")
        return pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance"])
    except Exception as e:
        print(f"Unexpected error opening PDF: {e}")
        return pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance"])
    df = basic_parse_text_to_df(text)
    df = standardize_columns(df)
    if 'Date' in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df["Date"] = df["Date"].fillna(pd.Timestamp("2025-01-01"))
    return df

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

def model_predict_and_explain(model, metrics_df):
    try:
        expected_features = [
            "Average Monthly Income", "Average Monthly Expenses", "Average Monthly EMI",
            "Net Surplus", "DTI Ratio", "Savings Rate", "Red Flag Count", "Credit Utilization",
            "Discretionary Expenses", "Income Stability"
        ]
        features = []
        for feature in expected_features:
            if feature in metrics_df.columns:
                value = metrics_df[feature].iloc[0]
                try:
                    features.append(float(value) if not pd.isna(value) else 0.0)
                except (ValueError, TypeError):
                    features.append(0.0)
            else:
                features.append(0.0)
        expected_feature_count = getattr(model, "n_features_in_", len(expected_features))
        if len(features) != expected_feature_count:
            features = features[:expected_feature_count] + [0.0] * (expected_feature_count - len(features))
        prediction = model.predict([features])[0]
        prob = model.predict_proba([features])[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
        return {
            "model_prediction": "APPROVED" if prediction == 1 else "REJECTED",
            "model_probability": round(max(prob) * 100, 2),
            "features_used": dict(zip(expected_features, features))
        }
    except Exception as e:
        print(f"[ERROR] ML model prediction failed: {e}")
        return None

def export_raw_csv(df, out_csv):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
    for col in ["Debit", "Credit", "Balance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')
    df["Description"] = df["Description"].astype(str)
    df.to_csv(out_csv, index=False, date_format='%Y-%m-%d')
    return out_csv

# Placeholder dependencies
def calculate_metrics(df, output_path):
    monthly = df.set_index("Date").groupby(pd.Grouper(freq="M")).agg(
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
        "Red Flag Count": int((df["Category"] == "Red Flags").sum()),
        "Credit Utilization": 0.0,
        "Discretionary Expenses": df[df["Category"] == "Discretionary Expenses"]["Debit"].sum(),
        "Income Stability": 0
    }])
    metrics_df.to_csv(output_path, index=False)
    return metrics_df

def identify_high_value_transactions(df):
    return df["Debit"].gt(df["Debit"].mean() + 2 * df["Debit"].std()) | \
           df["Credit"].gt(df["Credit"].mean() + 2 * df["Credit"].std())

def recurring_transactions(df):
    return df["Description"].value_counts().gt(2)[df["Description"]].values

def enforce_schema(df):
    return enforce_metrics_schema(df)

def make_arrow_compatible(df):
    return df

def score_and_decide(metrics_df, cibil_score, categorized_file):
    avg_income = metrics_df['Average Monthly Income'].iloc[0]
    red_flags = metrics_df['Red Flag Count'].iloc[0]
    if avg_income >= 30000 and red_flags == 0 and cibil_score >= 650:
        return {"Action": "Approve", "Reason": "Good income and credit profile", "Total Score": 80}
    return {"Action": "Reject", "Reason": "Insufficient income or credit issues", "Total Score": 40}

def extract_df_from_scanned_pdf(pdf_path):
    return pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance"])

# Placeholder plotting functions
def plot_income_trend_plotly(df):
    return None

def plot_surplus_trend_plotly(df):
    return None

def plot_income_vs_expenses_plotly(df):
    return None

def plot_cumulative_savings_plotly(df):
    return None

def plot_category_breakdown_plotly(df):
    return None

def plot_high_risk_timeline(df, output_path):
    return None

def plot_cibil_gauge(cibil_score, output_path):
    return None

def plot_credit_debit_ratio_trend_plotly(df):
    return None

def plot_category_expenses_over_time_plotly(df):
    return None

def plot_top_expenses_pareto_plotly(df):
    return None

def plot_income_expense_scatter_plotly(df):
    return None

def plot_monthly_transaction_volume_by_category_plotly(df):
    return None

def analyze_file(input_path, cibil_score=None, fill_method="interpolate", out_dir="outputs", applicant_data=None):
    out = {
        "raw_csv": None,
        "categorized_csv": None,
        "metrics_csv": None,
        "plots": [],
        "html_path": None,
        "pdf_path": None,
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

        if cibil_score is not None and (not isinstance(cibil_score, (int, float)) or cibil_score < 300 or cibil_score > 900):
            raise ValueError("CIBIL score must be a number between 300 and 900")
        cibil_score = cibil_score if cibil_score is not None else 700

        if applicant_data is None:
            applicant_data = {"name": Path(input_path).stem, "account_number": "Unknown"}
        else:
            applicant_data = {
                "name": applicant_data.get("name", Path(input_path).stem),
                "account_number": applicant_data.get("account_number", "Unknown")
            }

        print("[1/8] Extracting transactions...")
        df = None
        if input_path.suffix.lower() == ".pdf":
            df = parse_pdf_to_df(str(input_path))
        elif input_path.suffix.lower() in [".csv", ".txt", ".xls", ".xlsx"]:
            try:
                if input_path.suffix.lower() in [".xls", ".xlsx"]:
                    df = pd.read_excel(str(input_path))
                else:
                    df = pd.read_csv(str(input_path))
                df = standardize_columns(df)
            except Exception as e:
                print(f"File load error: {e}")
                df = pd.DataFrame()
        else:
            raise ValueError("Unsupported file type; supported: pdf, csv, xlsx, txt")

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01")).dt.strftime('%Y-%m-%d')
            df = df.dropna(subset=["Date"]).reset_index(drop=True)

        raw_csv_path = raw_dir / f"{input_path.stem}_raw.csv"
        export_raw_csv(df, raw_csv_path)
        out["raw_csv"] = str(raw_csv_path)

        df = standardize_columns(df)
        for col in ["Debit", "Credit", "Balance"]:
            if col not in df.columns:
                df[col] = 0.0
        if "Date" not in df.columns:
            print("No Date column found after parsing. Exiting.")
            return None

        print("[2/8] Categorizing transactions...")
        categorized_csv_path = categorized_dir / f"{input_path.stem}_categorized.csv"
        categorized_df = simple_categorize(df)
        categorized_df = standardize_columns(categorized_df)
        if "Date" in categorized_df.columns:
            categorized_df["Date"] = pd.to_datetime(categorized_df["Date"], errors="coerce").fillna(pd.Timestamp("2025-01-01"))
        categorized_df.to_csv(categorized_csv_path, index=False)
        out["categorized_csv"] = str(categorized_csv_path)

        print("[3/8] Calculating metrics...")
        metrics_file = metrics_dir / f"{input_path.stem}_metrics.csv"
        metrics_df = calculate_metrics(categorized_df, str(metrics_file))
        metrics_df = enforce_metrics_schema(metrics_df)
        out["metrics_csv"] = str(metrics_file)

        print("[4/8] Heuristic scoring & decisioning...")
        heuristic_decision = score_and_decide(metrics_df, cibil_score, str(categorized_csv_path))

        print("[5/8] ML model integration...")
        model_path = Path("models/loan_approval_model.pkl")
        ml_result = None
        if model_path.exists():
            model = load_model(model_path)
            if model:
                ml_result = model_predict_and_explain(model, metrics_df)

        print("[6/8] Final decision reconciliation...")
        final_action = heuristic_decision.get("Action", "Unknown")
        final_reason = heuristic_decision.get("Reason", "No reason provided")
        if ml_result and ml_result.get("model_probability", 0) >= 70:
            final_action = ml_result.get("model_prediction", "Unknown")
            final_reason = f"ML-based decision (confidence: {ml_result.get('model_probability', 0)}%)"

        print(f"Final Decision: {final_action}, Reason: {final_reason}")
        if st.runtime.exists():
            st.subheader("📋 Final Decision")
            if final_action.lower() == "approve":
                st.success(f"✅ {final_action} ({final_reason})")
            else:
                st.error(f"❌ {final_action} ({final_reason})")

        print("[7/8] Generating visualizations...")
        categorized_df["Amount"] = categorized_df["Credit"].fillna(0.0) - categorized_df["Debit"].fillna(0.0)
        plot_functions = [
            (plot_income_trend_plotly, "income_trend.png"),
            (plot_surplus_trend_plotly, "surplus_trend.png"),
            (plot_income_vs_expenses_plotly, "income_vs_expenses.png"),
        ]
        plot_paths = []
        @contextmanager
        def safe_browser():
            try:
                yield
            finally:
                try:
                    pio.kaleido.scope.stop()
                except Exception:
                    pass
        for func, filename in plot_functions:
            try:
                fig = func(categorized_df)
                if fig and st.runtime.exists():
                    st.plotly_chart(fig, use_container_width=True)
                output_path = str(plots_dir / filename)
                fig.write_image(output_path, engine="kaleido")
                plot_paths.append((output_path, filename.replace(".png", "").replace("_", " ").title()))
            except Exception as e:
                print(f"Error generating {filename}: {e}")
        out["plots"] = plot_paths

        print("[8/8] Building PDF report...")
        name = applicant_data.get('name', input_path.stem)
        if name.startswith('tmp') and '_' in name:
            name = name.split('_', 1)[-1]
        sanitized_name = re.sub(r'[^\w\s-]', '', name).replace(' ', '_').strip()
        pdf_path = reports_dir / f"{sanitized_name}_{cibil_score}_report.pdf"
        html_path = reports_dir / f"{sanitized_name}_{cibil_score}_report.html"
        premium_template = """<!DOCTYPE html><html><head><title>Loan Eligibility Report</title><style>body { font-family: Helvetica; margin: 40px; } .header { background-color: #f0f0f0; padding: 20px; border-radius: 8px; text-align: center; } .section { margin-top: 30px; } .table { width: 100%; border-collapse: collapse; margin-bottom: 20px; } .table th, .table td { border: 1px solid #ddd; padding: 12px; text-align: left; } .table th { background-color: #e0e0e0; font-weight: bold; } .table td { background-color: #f9f9f9; } h1 { color: #333; } h2 { color: #555; border-bottom: 2px solid #ccc; padding-bottom: 5px; } img { max-width: 100%; height: auto; display: block; margin: 10px 0; }</style></head><body><div class="header"><h1>Loan Eligibility Report</h1><p>Generated on: {timestamp}</p></div><div class="section"><h2>Applicant Data</h2><table class="table"><tr><th>Name</th><td>{applicant_name}</td></tr><tr><th>Account Number</th><td>{account_number}</td></tr><tr><th>CIBIL Score</th><td>{cibil_score}</td></tr></table></div><div class="section"><h2>Final Decision</h2><p style="color: {decision_color}; font-weight: bold;"><b>{final_action}</b>: {final_reason}</p></div><div class="section"><h2>Visualizations</h2>{plots_html}</div></body></html>"""
        def safe_float(value, default=0.0):
            try: return float(value)
            except: return default
        def safe_int(value, default=0):
            try: return int(value)
            except: return default
        def safe_str(value, default="Unknown"):
            try: return str(value).replace('{', '{{').replace('}', '}}').strip()
            except: return default
        avg_monthly_income = safe_float(metrics_df['Average Monthly Income'].iloc[0])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        applicant_name = safe_str(applicant_data.get('name'))
        account_number = safe_str(applicant_data.get('account_number'))
        decision_color = safe_str('#008000' if final_action.lower() == 'approve' else '#FF0000')
        plots_html = ''.join(f'<h3>{safe_str(title)}</h3><img src="data:image/png;base64,{base64.b64encode(open(path, "rb").read()).decode()}" alt="{safe_str(title)}"><br>' for path, title in plot_paths if os.path.exists(path))
        html_str = premium_template.format(
            timestamp=timestamp, applicant_name=applicant_name, account_number=account_number,
            cibil_score=cibil_score, avg_monthly_income=avg_monthly_income,
            final_action=final_action, final_reason=final_reason, decision_color=decision_color,
            plots_html=plots_html
        )
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_str)
        out["html_path"] = str(html_path)
        out["report_text"] = html_str

        try:
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name='HeaderTitle', fontName='Helvetica-Bold', fontSize=16, alignment=1))
            styles.add(ParagraphStyle(name='DecisionText', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor(decision_color)))
            story = [Table([[Paragraph("Loan Eligibility Report", styles['HeaderTitle'])], [Paragraph(f"Generated on: {timestamp}", styles['Normal'])]], colWidths=[A4[0]-80])]
            story.append(Spacer(1, 30))
            story.append(Paragraph("Applicant Data", styles['Heading2']))
            story.append(Table([[Paragraph("Name", styles['Normal']), Paragraph(applicant_name, styles['Normal'])],
                               [Paragraph("Account Number", styles['Normal']), Paragraph(account_number, styles['Normal'])],
                               [Paragraph("CIBIL Score", styles['Normal']), Paragraph(str(cibil_score), styles['Normal'])]], colWidths=[200, A4[0]-280]))
            story.append(Spacer(1, 30))
            story.append(Paragraph("Final Decision", styles['Heading2']))
            story.append(Paragraph(f"<b>{final_action}</b>: {final_reason}", styles['DecisionText']))
            story.append(Spacer(1, 30))
            for path, title in plot_paths:
                if os.path.exists(path):
                    story.append(Paragraph(title, styles['Heading3']))
                    story.append(Image(path, width=A4[0]-80, height=200))
                    story.append(Spacer(1, 10))
            doc.build(story)
            with open(pdf_path, 'wb') as f:
                f.write(pdf_buffer.getvalue())
            out["pdf_path"] = str(pdf_path)
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")

        print("[9/9] Saving decision JSON...")
        record = {
            "timestamp": timestamp,
            "file": str(input_path),
            "cibil": cibil_score,
            "action": final_action,
            "reason": final_reason
        }
        decision_json = decisions_dir / f"{input_path.stem}_decision.json"
        with open(decision_json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        out["decision_json"] = str(decision_json)

        if st.runtime.exists():
            st.markdown("### Loan Eligibility Report")
            st.markdown(f"**Generated on:** {timestamp}")
            st.markdown("#### Applicant Data")
            st.table({"Name": [applicant_name], "Account Number": [account_number], "CIBIL Score": [cibil_score]})
            st.markdown("#### Final Decision")
            if final_action.lower() == "approve":
                st.success(f"**{final_action}**: {final_reason}")
            else:
                st.error(f"**{final_action}**: {final_reason}")
            st.markdown("#### Visualizations")
            for path, title in plot_paths:
                if os.path.exists(path):
                    st.image(path, caption=title, use_container_width=True)
            if out["pdf_path"] and os.path.exists(out["pdf_path"]):
                with open(out["pdf_path"], "rb") as f:
                    st.download_button(label="📥 Download PDF", data=f, file_name=pdf_path.name, mime="application/pdf")

    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        print(f"Error in analysis: {e}")
    return out