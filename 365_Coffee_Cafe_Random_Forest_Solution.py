# 365 Coffee Cafe — Streamlit Repeat Visit Prediction
# Run: python -m streamlit run streamlit_app.py

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

RANDOM_STATE = 42
DATA_PATH = Path("365 Coffee Cafe Dataset.xlsx")

st.set_page_config(page_title="365 Coffee Cafe", page_icon="☕", layout="wide")
st.title("☕ 365 Coffee Café — Repeat Visit Prediction")
st.caption("Random Forest model to predict whether a customer returns within 14 days.")

@st.cache_data(show_spinner=False)
def load_workbook(uploaded_file):
    data = pd.read_excel(uploaded_file, sheet_name="Dataset")
    try:
        dictionary = pd.read_excel(uploaded_file, sheet_name="Data Dictionary")
    except Exception:
        dictionary = pd.DataFrame()
    return data, dictionary

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload 365 Coffee Cafe Dataset.xlsx", type=["xlsx", "xls"])
    page = st.radio("Page", ["Overview", "Business Analysis", "Model Performance", "Data Preview"])

try:
    if uploaded_file is not None:
        raw, dictionary = load_workbook(uploaded_file)
    elif DATA_PATH.exists():
        raw, dictionary = load_workbook(DATA_PATH)
    else:
        st.info("Upload **365 Coffee Cafe Dataset.xlsx** using the sidebar, or place it beside this script.")
        st.stop()
except Exception as e:
    st.error(f"Could not load workbook: {e}")
    st.stop()

required = [
    "basket_id", "transaction_date", "hour_of_day", "weekday", "store_location",
    "customer_age", "customer_gender", "loyalty_tier", "visit_number",
    "days_since_previous_visit", "order_channel", "payment_method", "promo_code",
    "discount_pct", "wait_time_min", "service_rating", "transaction_qty",
    "revenue", "discount_amount", "net_revenue", "transaction_id", "product_id",
    "product_category", "product_type", "returned_within_14_days"
]
missing = [c for c in required if c not in raw.columns]
if missing:
    st.error(f"Required columns missing from the Dataset sheet: {missing}")
    st.stop()

raw["transaction_date"] = pd.to_datetime(raw["transaction_date"], errors="coerce")

def mode_or_first(s):
    m = s.mode(dropna=True)
    return m.iloc[0] if not m.empty else s.iloc[0]

# Build one row per basket/visit, as required by the project.
g = raw.groupby("basket_id", sort=False)
visit = g.agg(
    transaction_date=("transaction_date", "first"),
    transaction_hour=("hour_of_day", "first"),
    weekday=("weekday", "first"),
    store_location=("store_location", "first"),
    customer_age=("customer_age", "first"),
    customer_gender=("customer_gender", "first"),
    loyalty_tier=("loyalty_tier", "first"),
    visit_number=("visit_number", "first"),
    days_since_previous_visit=("days_since_previous_visit", "first"),
    order_channel=("order_channel", "first"),
    payment_method=("payment_method", "first"),
    promo_code=("promo_code", "first"),
    discount_pct=("discount_pct", "first"),
    wait_time_min=("wait_time_min", "first"),
    service_rating=("service_rating", "first"),
    total_qty=("transaction_qty", "sum"),
    gross_revenue=("revenue", "sum"),
    total_discount=("discount_amount", "sum"),
    net_revenue=("net_revenue", "sum"),
    item_lines=("transaction_id", "count"),
    unique_products=("product_id", "nunique"),
    unique_categories=("product_category", "nunique"),
    unique_product_types=("product_type", "nunique"),
    target=("returned_within_14_days", "first"),
).reset_index()
visit["main_product_category"] = g["product_category"].agg(mode_or_first).values
visit["main_product_type"] = g["product_type"].agg(mode_or_first).values
visit["visit_month"] = visit["transaction_date"].dt.month
visit["day_of_month"] = visit["transaction_date"].dt.day
visit["day_of_week_num"] = visit["transaction_date"].dt.dayofweek
visit["is_weekend"] = visit["day_of_week_num"].isin([5, 6]).astype(int)
visit["discount_used"] = (pd.to_numeric(visit["discount_pct"], errors="coerce").fillna(0) > 0).astype(int)

target_map = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
visit["target_binary"] = visit["target"].astype(str).str.strip().str.lower().map(target_map)

features = [
    "transaction_hour", "weekday", "store_location", "customer_age",
    "customer_gender", "loyalty_tier", "visit_number", "days_since_previous_visit",
    "order_channel", "payment_method", "promo_code", "discount_pct",
    "wait_time_min", "service_rating", "total_qty", "gross_revenue",
    "total_discount", "net_revenue", "item_lines", "unique_products",
    "unique_categories", "unique_product_types", "main_product_category",
    "main_product_type", "visit_month", "day_of_month", "day_of_week_num",
    "is_weekend", "discount_used"
]

if page == "Overview":
    st.subheader("Business Overview")
    a, b, c, d = st.columns(4)
    a.metric("Transaction lines", f"{len(raw):,}")
    b.metric("Visits / baskets", f"{raw['basket_id'].nunique():,}")
    c.metric("Unique customers", f"{raw['customer_id'].nunique():,}" if "customer_id" in raw else "N/A")
    revenue = pd.to_numeric(raw["net_revenue"], errors="coerce").sum()
    d.metric("Net revenue (PKR)", f"{revenue:,.0f}")
    dates = raw["transaction_date"].dropna()
    if not dates.empty:
        st.write(f"**Date range:** {dates.min().date()} to {dates.max().date()}")
    q1, q2, q3 = st.columns(3)
    q1.metric("Missing values", f"{int(raw.isna().sum().sum()):,}")
    q2.metric("Duplicate rows", f"{int(raw.duplicated().sum()):,}")
    q3.metric("Modelling visits", f"{len(visit):,}")
    st.subheader("Data Dictionary")
    if not dictionary.empty:
        st.dataframe(dictionary, use_container_width=True, hide_index=True)
    st.subheader("Raw data preview")
    st.dataframe(raw.head(25), use_container_width=True, hide_index=True)

elif page == "Business Analysis":
    st.subheader("Business Analysis")
    branch = raw.groupby("store_location")["net_revenue"].sum().sort_values(ascending=False)
    category = raw.groupby("product_category")["net_revenue"].sum().sort_values(ascending=False)
    hour = raw.groupby("hour_of_day")["net_revenue"].sum().sort_index()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = raw.groupby("weekday")["net_revenue"].sum()
    weekday = weekday.reindex([x for x in weekday_order if x in weekday.index])

    left, right = st.columns(2)
    with left:
        st.markdown("#### Net Revenue by Branch (PKR)")
        st.bar_chart(branch)
    with right:
        st.markdown("#### Net Revenue by Product Category (PKR)")
        st.bar_chart(category)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Net Revenue by Trading Hour")
        st.line_chart(hour)
    with right:
        st.markdown("#### Net Revenue by Weekday")
        st.bar_chart(weekday)
    a, b = st.columns(2)
    a.metric("Average wait time", f"{pd.to_numeric(raw['wait_time_min'], errors='coerce').mean():.2f} min")
    b.metric("Average service rating", f"{pd.to_numeric(raw['service_rating'], errors='coerce').mean():.2f} / 5")
    st.info("These are descriptive patterns and do not prove causation.")

elif page == "Model Performance":
    st.subheader("Random Forest Model Performance")
    data = visit.dropna(subset=["target_binary"]).copy()
    data["target_binary"] = data["target_binary"].astype(int)
    if data["target_binary"].nunique() < 2 or data["target_binary"].value_counts().min() < 2:
        st.error("The target must contain at least two examples of both Yes and No.")
        st.stop()

    X, y = data[features].copy(), data["target_binary"]
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [col for col in X.columns if col not in cat_cols]
    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=2
        ))
    ])
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
        )
        with st.spinner("Training Random Forest..."):
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        st.error(f"Could not train the model: {e}")
        st.stop()

    auc = roc_auc_score(y_test, prob) if y_test.nunique() == 2 else float("nan")
    a, b, c, d, e = st.columns(5)
    a.metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
    b.metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
    c.metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.3f}")
    d.metric("F1-score", f"{f1_score(y_test, pred, zero_division=0):.3f}")
    e.metric("ROC-AUC", f"{auc:.3f}" if np.isfinite(auc) else "N/A")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred, labels=[0, 1]),
        display_labels=["No return", "Return"]
    ).plot(ax=ax, values_format="d", colorbar=False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")
    report = classification_report(
        y_test, pred, labels=[0, 1], target_names=["No return", "Return"],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    st.subheader("Top Feature Importances")
    try:
        names = model.named_steps["preprocessor"].get_feature_names_out()
        importance = model.named_steps["model"].feature_importances_
        imp = pd.DataFrame({"Feature": names, "Importance": importance})
        imp = imp.sort_values("Importance", ascending=False).head(15)
        st.bar_chart(imp.set_index("Feature"))
        st.dataframe(imp, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Feature importance could not be displayed: {e}")
    st.caption("Feature importance reflects model associations, not proof of causation.")

elif page == "Data Preview":
    st.subheader("Raw Transaction Data")
    st.dataframe(raw, use_container_width=True, hide_index=True)
    st.subheader("Visit-Level Modelling Data")
    st.dataframe(visit, use_container_width=True, hide_index=True)
    csv = visit.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download visit-level dataset (CSV)",
        data=csv,
        file_name="365_Coffee_Cafe_visit_level_modeling_dataset.csv",
        mime="text/csv"
    )
