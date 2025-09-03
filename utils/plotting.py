import io
import base64
import pandas as pd
import numpy as np
import matplotlib as plt
import plotly.express as px
import plotly.graph_objects as go

def save_fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

def fill_months_and_plot_timeseries(df, date_col, value_col, out_png, fill_method="zero", title=None, ylabel=None):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
    df = df.dropna(subset=[date_col])
    if df.empty:
        return None, None
    monthly = df.groupby(df[date_col].dt.to_period("M"))[value_col].sum()
    full_range = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(full_range).astype(float)
    if fill_method == "zero":
        monthly = monthly.fillna(0.0)
    else:
        monthly = monthly.interpolate().fillna(0.0)
    if isinstance(monthly.index, pd.PeriodIndex):
        x = monthly.index.to_timestamp()
    else:
        x = pd.to_datetime(monthly.index)
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(x, monthly.values, marker='o', linewidth=2)
    ax.bar(x, monthly.values, alpha=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels([pd.Timestamp(dt).strftime("%b %Y") for dt in x], rotation=45)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title(title or f"Monthly {value_col}")
    ax.set_ylabel(ylabel or value_col)
    fig.tight_layout()
    fig.savefig(out_png)
    b64 = save_fig_to_base64(fig)
    return out_png, b64

def plot_credit_debit_ratio_trend_plotly(df):
    try:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df_monthly = df.groupby(df['Date'].dt.to_period('M')).agg({'Credit': 'sum', 'Debit': 'sum'}).reset_index()
        df_monthly['Ratio'] = df_monthly['Credit'] / df_monthly['Debit'].replace(0, np.nan)
        df_monthly['Date'] = df_monthly['Date'].dt.to_timestamp()
        fig = px.line(df_monthly, x='Date', y='Ratio', title='Credit/Debit Ratio Trend', markers=True)
        fig.update_traces(line=dict(color='#FF6384', width=3))
        fig.update_layout(yaxis_title='Credit/Debit Ratio', hovermode='x unified')
        return fig
    except Exception as e:
        print(f"Error in plot_credit_debit_ratio_trend_plotly: {e}")
        return None

def plot_category_expenses_over_time_plotly(df):
    try:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df_monthly = df.groupby([df['Date'].dt.to_period('M'), 'Category'])['Debit'].sum().reset_index()
        df_monthly['Date'] = df_monthly['Date'].dt.to_timestamp()
        fig = px.area(df_monthly, x='Date', y='Debit', color='Category', title='Expenses by Category Over Time')
        fig.update_layout(yaxis_title='Expenses (₹)', hovermode='x unified')
        return fig
    except Exception as e:
        print(f"Error in plot_category_expenses_over_time_plotly: {e}")
        return None

def plot_top_expenses_pareto_plotly(df):
    try:
        df = df.copy()
        top_expenses = df.groupby('Description')['Debit'].sum().sort_values(ascending=False).head(10).reset_index()
        top_expenses['Cumulative'] = top_expenses['Debit'].cumsum() / top_expenses['Debit'].sum() * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_expenses['Description'],
            y=top_expenses['Debit'],
            name='Expenses',
            marker_color='#FF6384'
        ))
        fig.add_trace(go.Scatter(
            x=top_expenses['Description'],
            y=top_expenses['Cumulative'],
            name='Cumulative %',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='#36A2EB')
        ))
        fig.update_layout(
            title='Top 10 Expenses (Pareto Chart)',
            yaxis=dict(title='Expenses (₹)'),
            yaxis2=dict(title='Cumulative Percentage (%)', overlaying='y', side='right', range=[0, 100]),
            xaxis_tickangle=45
        )
        return fig
    except Exception as e:
        print(f"Error in plot_top_expenses_pareto_plotly: {e}")
        return None

def plot_income_expense_scatter_plotly(df):
    try:
        df = df.copy()
        df['Amount'] = df['Credit'].fillna(0.0) - df['Debit'].fillna(0.0)
        fig = px.scatter(
            df, x='Credit', y='Debit', color='Category',
            size=df['Amount'].abs(), title='Income vs Expense Scatter Plot',
            hover_data=['Description', 'Date']
        )
        fig.update_layout(xaxis_title='Income (₹)', yaxis_title='Expenses (₹)', hovermode='closest')
        return fig
    except Exception as e:
        print(f"Error in plot_income_expense_scatter_plotly: {e}")
        return None

def plot_monthly_transaction_volume_by_category_plotly(df):
    try:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df_monthly = df.groupby([df['Date'].dt.to_period('M'), 'Category']).size().reset_index(name='Count')
        df_monthly['Date'] = df_monthly['Date'].dt.to_timestamp()
        fig = px.bar(df_monthly, x='Date', y='Count', color='Category', title='Monthly Transaction Volume by Category', barmode='stack')
        fig.update_layout(yaxis_title='Transaction Count', hovermode='x unified')
        return fig
    except Exception as e:
        print(f"Error in plot_monthly_transaction_volume_by_category_plotly: {e}")
        return None
    
def plot_income_vs_expense_from_df(df, out_png):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
    df = df.dropna(subset=["Date"])
    monthly = df.set_index("Date").groupby(pd.Grouper(freq="ME")).agg(Income=("Credit","sum"), Expenses=("Debit","sum"))
    if monthly.empty:
        return None, None
    if isinstance(monthly.index, pd.PeriodIndex):
        x = monthly.index.to_timestamp()
    else:
        x = pd.to_datetime(monthly.index)
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(x - pd.Timedelta(days=6), monthly["Income"], width=12, label="Income")
    ax.bar(x + pd.Timedelta(days=6), monthly["Expenses"], width=12, label="Expenses")
    ax.set_xticks(x)
    ax.set_xticklabels([pd.Timestamp(d).strftime("%b %Y") for d in x], rotation=45)
    ax.set_title("Income vs Expenses (Monthly)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png)
    b64 = save_fig_to_base64(fig)
    return out_png, b64

def plot_category_pie(df, out_png):
    if "Category" not in df.columns:
        return None, None
    counts = df["Category"].value_counts()
    if counts.empty:
        return None, None
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
    ax.set_title("Spending by Category")
    fig.tight_layout()
    fig.savefig(out_png)
    b64 = save_fig_to_base64(fig)
    return out_png, b64

def plot_high_risk_timeline(df, out_png):
    try:
        if "Category" not in df.columns:
            return None
        high = df[df["Category"].str.lower() == "red flags"].copy()
        if high.empty:
            return None
        high["Date"] = pd.to_datetime(high["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
        high = high.dropna(subset=["Date"])
        fig = px.scatter(high, x="Date", y=[1]*len(high), text="Description",
                         title="High-Risk Transactions Timeline (Red Flags)")
        fig.update_traces(marker=dict(size=10, symbol="x", color="red"), textposition="top center")
        fig.update_yaxes(showticklabels=False)
        fig.write_image(out_png)
        return fig
    except Exception as e:
        print(f"Error in plot_high_risk_timeline: {e}")
        return None
    
def plot_cibil_gauge(cibil_score, output_path):
    try:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cibil_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CIBIL Score"},
            gauge={
                'axis': {'range': [300, 900]},
                'bar': {'color': "#FF6384"},
                'steps': [
                    {'range': [300, 600], 'color': "red"},
                    {'range': [600, 750], 'color': "yellow"},
                    {'range': [750, 900], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': cibil_score
                }
            }
        ))
        fig.write_image(output_path)
        return fig
    except Exception as e:
        print(f"Error in plot_cibil_gauge: {e}")
        return None


def plot_income_trend_plotly(df):
    try:
        df = df.copy()
        if "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
        if "Credit" not in df.columns:
            return None
        monthly_income = df.groupby(pd.Grouper(key="Date", freq="ME"))["Credit"].sum().reset_index()
        fig = px.line(monthly_income, x="Date", y="Credit", markers=True,
                      title="Monthly Income Trend", labels={"Credit": "Income (₹)"})
        fig.update_traces(line=dict(width=3))
        return fig
    except Exception as e:
        print(f"Error in plot_income_trend_plotly: {e}")
        return None

def plot_surplus_trend_plotly(df):
    try:
        df = df.copy()
        if "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
        if not {"Credit", "Debit"}.issubset(df.columns):
            return None
        df["Surplus"] = df["Credit"] - df["Debit"]
        monthly_surplus = df.groupby(pd.Grouper(key="Date", freq="ME"))["Surplus"].sum().reset_index()
        fig = px.line(monthly_surplus, x="Date", y="Surplus", markers=True,
                      title="Monthly Surplus Trend", labels={"Surplus": "Surplus (₹)"})
        fig.update_traces(line=dict(width=3))
        return fig
    except Exception as e:
        print(f"Error in plot_surplus_trend_plotly: {e}")
        return None

def plot_income_vs_expenses_plotly(df):
    try:
        df = df.copy()
        if "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
        if not {"Credit", "Debit"}.issubset(df.columns):
            return None
        monthly = df.groupby(pd.Grouper(key="Date", freq="ME")).agg(
            Income=("Credit", "sum"),
            Expenses=("Debit", "sum")
        ).reset_index()
        monthly_long = monthly.melt(id_vars="Date", value_vars=["Income", "Expenses"],
                                   var_name="Type", value_name="Amount")
        fig = px.bar(monthly_long, x="Date", y="Amount", color="Type", barmode="group",
                     title="Monthly Income vs Expenses", labels={"Amount": "Amount (₹)"})
        return fig
    except Exception as e:
        print(f"Error in plot_income_vs_expenses_plotly: {e}")
        return None

def plot_cumulative_savings_plotly(df):
    try:
        df = df.copy()
        if "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
        if not {"Credit", "Debit"}.issubset(df.columns):
            return None
        df["Savings"] = df["Credit"] - df["Debit"]
        df = df.sort_values("Date")
        df["Cumulative Savings"] = df["Savings"].cumsum()
        fig = px.line(df, x="Date", y="Cumulative Savings", markers=True,
                      title="Cumulative Savings Over Time", labels={"Cumulative Savings": "Savings (₹)"})
        fig.update_traces(line=dict(width=3))
        return fig
    except Exception as e:
        print(f"Error in plot_cumulative_savings_plotly: {e}")
        return None

def plot_category_breakdown_plotly(df):
    try:
        if "Category" not in df.columns:
            return None
        cat_counts = df["Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.pie(cat_counts, values="Count", names="Category",
                     title="Spending by Category", hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return fig
    except Exception as e:
        print(f"Error in plot_category_breakdown_plotly: {e}")
        return None