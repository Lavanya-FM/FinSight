import os
import pickle
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

# Configure logging for real-time monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------
# Part 1: Transaction Categorization Model (Unchanged)
# -------------------------------

def train_transaction_model():
    try:
        logger.info("Starting transaction categorization model training")
        
        transaction_data = pd.DataFrame({
            "Description": [
                "Salary Credit from Employer", "Rent Payment via NEFT", "Home Loan EMI Deduction",
                "Grocery Shopping at Supermarket", "Electricity Bill Payment", "Car Loan Repayment",
                "Payroll Deposit", "ATM Cash Withdrawal", "Credit Card Payment",
                "Interest Credited to Savings Account", "Mobile Recharge", "Loan Disbursement",
                "Dividend Received", "Fuel Purchase", "Insurance Premium Payment",
                "Freelance Payment Received", "Dining Out Expense", "Personal Loan EMI",
                "Refund from Online Shopping", "Water Bill Payment", "Education Loan Repayment",
                "Bonus Credit", "Movie Tickets Purchase", "Medical Bill Payment",
                "Investment Maturity Proceeds", "Travel Booking Expense", "Credit Card Annual Fee",
                "Pension Deposit", "Gym Membership Fee", "Vehicle Loan EMI",
                "Gift Received via UPI", "Clothing Purchase", "Home Renovation Loan Payment",
                "Tax Refund", "Subscription Service Fee", "Overdraft Fee",
                "Consulting Fee Received", "Utility Bill Auto-Debit", "Mortgage Payment",
                "Interest on Fixed Deposit", "Online Shopping Delivery Charge", "Student Loan EMI",
                "Royalty Payment Received", "Repair and Maintenance Expense", "Business Loan Repayment",
                "Gratuity Credit", "Entertainment Subscription", "Late Payment Fee",
                "Wage Deposit", "Household Supplies Purchase", "Gold Loan EMI",
                "Monthly Salary Transfer", "Apartment Rent Debit", "EMI for Personal Loan",
                "Supermarket Grocery Bill", "Phone Bill Payment", "Bike Loan Installment",
                "Direct Deposit Payroll", "Cash Withdrawal from ATM", "Minimum Due on Credit Card",
                "Savings Interest Credit", "Prepaid Mobile Top-Up", "New Loan Amount Credited",
                "Stock Dividend Deposit", "Petrol Station Payment", "Life Insurance Premium",
                "Gig Economy Earnings", "Restaurant Bill Payment", "EMI Bounce Charge",
                "E-commerce Refund", "Internet Bill Payment", "Study Abroad Loan Repayment",
                "Performance Bonus", "Cinema Ticket Booking", "Hospital Visit Expense",
                "Mutual Fund Redemption", "Flight Ticket Purchase", "Card Maintenance Fee",
                "Retirement Pension", "Fitness Club Subscription", "Auto Loan Payment",
                "UPI Gift Received", "Apparel Shopping", "Construction Loan EMI",
                "Income Tax Refund", "Streaming Service Fee", "Bank Overdraft Charge",
                "Professional Fee Credit", "Gas Bill Auto-Pay", "Property Mortgage EMI",
                "FD Interest Payout", "Courier Service Charge", "Education EMI Deduction",
                "Book Royalty Received", "Car Repair Bill", "SME Loan Repayment",
                "End-of-Service Gratuity", "Music Streaming Fee", "Penalty for Late Payment",
                "Hourly Wage Deposit", "Groceries Delivery", "Jewelry Loan EMI"
            ],
            "Category": [
                "Income", "Expense", "Debt", "Expense", "Expense", "Debt", "Income", "Expense", "Debt",
                "Income", "Expense", "Income", "Income", "Expense", "Expense", "Income", "Expense", "Debt",
                "Income", "Expense", "Debt", "Income", "Expense", "Expense", "Income", "Expense", "Expense",
                "Income", "Expense", "Debt", "Income", "Expense", "Debt", "Income", "Expense", "Expense",
                "Income", "Expense", "Debt", "Income", "Expense", "Debt", "Income", "Expense", "Debt",
                "Income", "Expense", "Expense", "Income", "Expense", "Debt",
                "Income", "Expense", "Debt", "Expense", "Expense", "Debt", "Income", "Expense", "Debt",
                "Income", "Expense", "Income", "Income", "Expense", "Expense", "Income", "Expense", "Expense",
                "Income", "Expense", "Debt", "Income", "Expense", "Expense", "Income", "Expense", "Expense",
                "Income", "Expense", "Debt", "Income", "Expense", "Debt", "Income", "Expense", "Expense",
                "Income", "Expense", "Debt", "Income", "Expense", "Debt", "Income", "Expense", "Debt",
                "Income", "Expense", "Expense", "Income", "Expense", "Debt"
            ]
        })

        if transaction_data.empty or transaction_data["Description"].isnull().any():
            logger.error("Transaction data is empty or contains null values")
            raise ValueError("Invalid transaction data")

        vectorizer = TfidfVectorizer(max_features=500)
        X_trans = vectorizer.fit_transform(transaction_data["Description"])
        logger.info(f"Vectorized {X_trans.shape[0]} transactions with {X_trans.shape[1]} features")

        category_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        category_classifier.fit(X_trans, transaction_data["Category"])
        logger.info("Transaction categorization model trained")

        os.makedirs("models", exist_ok=True)
        with open("models/vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f, protocol=4)
        with open("models/classifier.pkl", "wb") as f:
            pickle.dump(category_classifier, f, protocol=4)
        with open("models/label_encoder.pkl", "wb") as f:
            pickle.dump(None, f, protocol=4)
        logger.info("Transaction categorization models saved")

    except Exception as e:
        logger.error(f"Error in transaction model training: {str(e)}")
        raise

# -------------------------------
# Part 2: Loan Approval Model
# -------------------------------

def train_loan_model():
    try:
        logger.info("Starting loan approval model training")
        start_time = time.time()

        # Load Kaggle Lending Club dataset with subsampling to manage memory
        data_path = r"C:\Users\lavan\Downloads\archive (17)\accepted_2007_to_2018Q4.csv.gz"  # Update path if needed
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path, nrows=100000, low_memory=False)

        # Select relevant features
        selected_features = [
            'annual_inc', 'dti', 'revol_util', 'emp_length', 'inq_last_6mths', 'delinq_2yrs',
            'loan_amnt', 'open_acc', 'total_acc', 'pub_rec', 'int_rate', 'revol_bal',
            'fico_range_low', 'fico_range_high', 'term', 'grade', 'home_ownership',
            'verification_status', 'loan_status'
        ]
        df = df[selected_features].copy()

        # Preprocess data
        # Handle employment length
        df['emp_length'] = df['emp_length'].replace({
            '< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
            '10+ years': 10
        }).astype(float)

        # Handle revol_util and int_rate (convert to string, remove %, convert to float)
        df['revol_util'] = df['revol_util'].astype(str).replace('nan', '0').str.replace('%', '').astype(float)
        df['int_rate'] = df['int_rate'].astype(str).replace('nan', '0').str.replace('%', '').astype(float)

        # Compute credit_score
        df['credit_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2
        df = df.drop(['fico_range_low', 'fico_range_high'], axis=1)

        # Compute loan_to_income
        df['loan_to_income'] = df['loan_amnt'] / df['annual_inc'].replace(0, 1)

        # Approximate missing features
        df['net_surplus'] = df['annual_inc'] * 0.18
        df['savings_rate'] = np.random.normal(18, 8, len(df)).clip(0, 50)
        df['income_trend'] = np.random.uniform(-0.15, 0.25, len(df))
        df['emi_bounce_count'] = df['delinq_2yrs'].clip(0, 4)
        df['income_stability'] = np.random.choice([0, 1], size=len(df), p=[0.3, 0.7])
        df['discretionary_expenses'] = np.random.normal(72000, 30000, len(df)).clip(0, 500000)
        df['dependents'] = np.random.poisson(1.2, len(df)).clip(0, 4)
        df['debt_consolidation'] = (df['loan_amnt'] > 0).astype(int)

        # Create and validate label
        print("Unique loan_status values:", df['loan_status'].unique())  # Debug: Check values
        df['label'] = df['loan_status'].apply(lambda x: 1 if x == 'Fully Paid' else 0 if x in ['Charged Off', 'Default'] else np.nan)
        df = df.dropna(subset=['label'])  # Drop rows where label is NaN
        logger.info(f"After dropping NaN labels, {df.shape[0]} rows remain")
        print("Columns after label creation:", df.columns.tolist())  # Debug: Check columns

        # Subsample to 100,000 rows (already done with nrows, but ensure consistency)
        # df = df.sample(n=100000, random_state=42)  # Uncomment if needed
        logger.info(f"Subsampled to {df.shape[0]} rows with {df.shape[1]-1} features")

        # Handle categorical features
        categorical_features = ['term', 'grade', 'home_ownership', 'verification_status']
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_features = encoder.fit_transform(df[categorical_features])
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)

        # Combine numeric and encoded categorical features, preserve label
        numeric_features = [
            'annual_inc', 'dti', 'revol_util', 'emp_length', 'inq_last_6mths', 'delinq_2yrs',
            'loan_amnt', 'open_acc', 'total_acc', 'pub_rec', 'int_rate', 'revol_bal',
            'credit_score', 'loan_to_income', 'net_surplus', 'savings_rate', 'income_trend',
            'emi_bounce_count', 'income_stability', 'discretionary_expenses', 'dependents',
            'debt_consolidation'
        ]
        df_numeric = df[numeric_features]
        df = pd.concat([df_numeric, df['label'], encoded_df], axis=1)  # Explicitly include label

        # Handle missing values (only for features, not label)
        feature_columns = [col for col in df.columns if col != 'label']
        df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean(numeric_only=True))

        # Validate data (check features only)
        if df.empty or df[feature_columns].isnull().any().any():
            logger.error("Loan feature data contains null values")
            raise ValueError("Invalid loan feature data")

        # Feature Engineering
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(df.drop('label', axis=1))
        poly_feature_names = poly.get_feature_names_out(df.drop('label', axis=1).columns)
        X = pd.DataFrame(X_poly, columns=poly_feature_names)
        y = df['label']
        logger.info(f"Generated {X.shape[1]} features after polynomial transformation")

        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Features scaled")

        # Create Holdout Set
        X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
        logger.info(f"Split data: {X_train_val.shape[0]} training/validation, {X_holdout.shape[0]} holdout")

        # Feature Selection with RFE
        rfe_selector = RFE(estimator=RandomForestClassifier(n_jobs=-1), n_features_to_select=10, step=1)  # Reduced to 10 features
        rfe_selector.fit(X_train_val, y_train_val)
        selected_features = X.columns[rfe_selector.support_]
        logger.info(f"Selected {len(selected_features)} features: {selected_features.tolist()}")

        X_train_val_selected = X_train_val[:, rfe_selector.support_]
        X_holdout_selected = X_holdout[:, rfe_selector.support_]

        # Evaluate Models
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_jobs=-1),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
        }

        logger.info("Cross-validation Results (10-fold, F1 Score):")
        for name, model in models.items():
            scores = cross_val_score(model, X_train_val_selected, y_train_val, cv=10, scoring='f1')
            logger.info(f"{name}: F1 = {scores.mean():.3f} ± {scores.std():.3f}")

        # Grid Search for RandomForest
        param_grid = {
            "n_estimators": [100],  # Reduced options for speed
            "max_depth": [10, None],
            "min_samples_split": [2],
            "max_features": ['sqrt']
        }
        grid_search = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train_val_selected, y_train_val)
        best_model = grid_search.best_estimator_
        logger.info(f"Best RandomForest Params: {grid_search.best_params_}")

        # Final Evaluation on Test Set
        X_train, X_test, y_train, y_test = train_test_split(X_train_val_selected, y_train_val, test_size=0.2, random_state=42)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        logger.info("\nClassification Report (Test Set):")
        logger.info("\n" + classification_report(y_test, y_pred))

        # Holdout Set Evaluation
        y_holdout_pred = best_model.predict(X_holdout_selected)
        logger.info("\nClassification Report (Holdout Set):")
        logger.info("\n" + classification_report(y_holdout, y_holdout_pred))

        # Feature Importance
        feature_importance_df = pd.DataFrame({
            "Feature": selected_features,
            "Importance": best_model.feature_importances_
        })
        logger.info("\nFeature Importance:")
        logger.info("\n" + feature_importance_df.sort_values(by="Importance", ascending=False).to_string())

        # Visualizations
        os.makedirs("plots", exist_ok=True)
        
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.title("Confusion Matrix (Test Set)")
        plt.savefig("plots/confusion_matrix.png")
        plt.close()
        logger.info("Confusion matrix saved to plots/confusion_matrix.png")

        RocCurveDisplay.from_estimator(best_model, X_test, y_test)
        plt.title("ROC Curve (Test Set)")
        plt.savefig("plots/roc_curve.png")
        plt.close()
        logger.info("ROC curve saved to plots/roc_curve.png")

        feature_importance_df.sort_values(by="Importance", ascending=False).plot(
            x='Feature', y='Importance', kind='bar', figsize=(10, 6)
        )
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png")
        plt.close()
        logger.info("Feature importance plot saved to plots/feature_importance.png")

        # Save Models and Preprocessors
        with open("models/loan_approval_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open("models/poly_features.pkl", "wb") as f:
            pickle.dump(poly, f)
        with open("models/rfe_selector.pkl", "wb") as f:
            pickle.dump(rfe_selector, f)
        with open("models/encoder.pkl", "wb") as f:
            pickle.dump(None, f)  # No encoder needed here as categoricals handled separately
        logger.info("Loan approval models and preprocessors saved")

        training_time = time.time() - start_time
        logger.info(f"Loan model training completed in {training_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error in loan model training: {str(e)}")
        raise

# Main execution
if __name__ == "__main__":
    try:
        train_transaction_model()
        train_loan_model()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise