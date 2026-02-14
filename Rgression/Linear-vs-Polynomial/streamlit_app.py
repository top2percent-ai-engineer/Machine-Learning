import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import altair as alt
import os

# --- Cached CSV reader (no widgets inside) ---
@st.cache_data
def read_csv_file(path_or_buffer):
    """Read CSV safely and cache it."""
    return pd.read_csv(path_or_buffer)

# --- Cached model trainer ---
@st.cache_resource
def train_models(df: pd.DataFrame, poly_degree: int = 2):
    X = df[["Level"]]
    y = df["Salary"]

    # Linear Regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_pred_lin = lin_model.predict(X_test)
    r2_lin = r2_score(y_test, y_pred_lin)

    # Polynomial Regression (fit on full data)
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    y_pred_poly = poly_model.predict(X_poly)
    r2_poly = r2_score(y, y_pred_poly)

    return {
        "lin_model": lin_model,
        "poly_model": poly_model,
        "poly_transformer": poly,
        "r2_lin": r2_lin,
        "r2_poly": r2_poly,
    }

# --- Main app ---
def main():
    st.set_page_config(page_title="Position Salary Predictor", page_icon="💼", layout="centered")
    st.title("💼 Position Salary Predictor")
    st.markdown(
        """
This app predicts salary based on position level using:

- Linear Regression
- Polynomial Regression
"""
    )

    # --- Sidebar ---
    st.sidebar.header("Configuration")
    poly_degree = st.sidebar.slider(
        "Polynomial degree", min_value=2, max_value=6, value=2, step=1
    )

    # --- File upload or default CSV ---
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (optional)",
        type=["csv"],
        help="Must contain columns: Position, Level, Salary"
    )

    if uploaded_file is not None:
        df = read_csv_file(uploaded_file)
    else:
        default_csv = "Position_Salaries.csv"
        if os.path.exists(default_csv):
            df = read_csv_file(default_csv)
        else:
            st.warning("No default CSV found. Please upload a CSV file!")
            st.stop()  # Stop execution until user uploads

    # --- Validate columns ---
    expected_cols = {"Position", "Level", "Salary"}
    if not expected_cols.issubset(df.columns):
        st.error(f"CSV must contain columns: {', '.join(expected_cols)}")
        st.stop()

    # --- EDA ---
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Shape**")
        st.write(df.shape)
    with col2:
        st.markdown("**Missing Values**")
        st.write(df.isnull().sum())

    with st.expander("Show descriptive statistics"):
        st.write(df.describe())

    # --- Train models ---
    models = train_models(df, poly_degree=poly_degree)

    st.subheader("📈 Model Performance")
    st.markdown(
        f"""
**Linear Regression R² (test set)**: `{models['r2_lin']:.4f}`  
**Polynomial Regression R² (degree {poly_degree})**: `{models['r2_poly']:.4f}`
"""
    )

    # --- Prediction ---
    model_choice = st.radio(
        "Select model for prediction",
        ["Linear Regression", "Polynomial Regression"],
        index=1 if models["r2_poly"] >= models["r2_lin"] else 0,
        horizontal=True,
    )

    st.subheader("🔮 Predict Salary by Level")
    min_level = int(df["Level"].min())
    max_level = int(df["Level"].max())
    level_input = st.number_input(
        "Enter position level",
        min_value=float(min_level),
        max_value=float(max_level + 2),
        value=float(min_level),
        step=1.0,
    )

    if st.button("Predict Salary"):
        level_array = np.array([[level_input]])
        if model_choice == "Linear Regression":
            pred = models["lin_model"].predict(level_array)[0]
        else:
            level_poly = models["poly_transformer"].transform(level_array)
            pred = models["poly_model"].predict(level_poly)[0]
        st.success(f"Estimated salary for level {level_input:.0f}: **₹{pred:,.2f}**")

    # --- Visualization ---
    st.subheader("📉 Regression Visualization")
    X = df[["Level"]].values
    y = df["Salary"].values
    X_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    if model_choice == "Linear Regression":
        y_grid_pred = models["lin_model"].predict(X_grid)
        title = "Linear Regression"
    else:
        X_grid_poly = models["poly_transformer"].transform(X_grid)
        y_grid_pred = models["poly_model"].predict(X_grid_poly)
        title = f"Polynomial Regression (degree {poly_degree})"

    chart_df = pd.DataFrame({"Level": X.flatten(), "Actual Salary": y})
    curve_df = pd.DataFrame({"Level": X_grid.flatten(), "Predicted Salary": y_grid_pred})

    scatter = alt.Chart(chart_df).mark_circle(size=80, color="steelblue").encode(
        x="Level", y="Actual Salary"
    )
    line = alt.Chart(curve_df).mark_line(color="red").encode(
        x="Level", y="Predicted Salary"
    )
    st.altair_chart((scatter + line).properties(title=title), use_container_width=True)

if __name__ == "__main__":
    main()
