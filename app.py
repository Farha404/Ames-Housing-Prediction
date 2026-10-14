from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st  # type: ignore[import-not-found]

from model.project_runner import ARTIFACT, DATASET, load_dataset, load_model, notebook_feature_schema

st.set_page_config(
    page_title="Ames Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
    .hero {padding: 1.2rem 1.4rem; border: 1px solid rgba(128,128,128,.25); border-radius: 16px; margin-bottom: 1.2rem;}
    .price {font-size: 2.7rem; font-weight: 750; margin: .25rem 0;}
    .muted {opacity: .72;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("🏠 Ames Housing")
st.sidebar.caption("Machine-learning price prediction")
page = st.sidebar.radio(
    "Navigation",
    ["Price Prediction", "Model Performance", "Feature Importance", "Model Information", "About"],
)

@st.cache_resource(show_spinner="Loading the final SVR model…")
def cached_model():
    return load_model()

@st.cache_data(show_spinner=False)
def cached_data():
    return load_dataset()

@st.cache_data(show_spinner=False)
def cached_schema():
    return notebook_feature_schema()


def ensure_project_ready():
    if not DATASET.exists():
        st.error("AmesHousing.csv is missing. Place it in the project root.")
        st.stop()


def input_value(df: pd.DataFrame, feature: str):
    series = df[feature]
    if pd.api.types.is_numeric_dtype(series):
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return 0
        median = float(clean.median())
        if pd.api.types.is_integer_dtype(series):
            return int(round(median))
        return median
    mode = series.dropna().mode()
    return str(mode.iloc[0]) if not mode.empty else ""


def numeric_widget(df, feature, key_prefix):
    s = pd.to_numeric(df[feature], errors="coerce").dropna()
    default = input_value(df, feature)
    if s.empty:
        return st.number_input(feature, value=0.0, key=f"{key_prefix}_{feature}")
    lo, hi = float(s.min()), float(s.max())
    if pd.api.types.is_integer_dtype(df[feature]):
        return st.number_input(
            feature,
            min_value=int(np.floor(lo)),
            max_value=int(np.ceil(hi)),
            value=int(np.clip(default, lo, hi)),
            step=1,
            key=f"{key_prefix}_{feature}",
        )
    return st.number_input(
        feature,
        min_value=float(lo),
        max_value=float(hi),
        value=float(np.clip(default, lo, hi)),
        key=f"{key_prefix}_{feature}",
    )


def categorical_widget(df, feature, options, key_prefix):
    if feature in options:
        choices = options[feature]
    else:
        choices = [str(x) for x in df[feature].dropna().unique().tolist()]
    if not choices:
        choices = ["NA"]
    default = str(input_value(df, feature))
    index = choices.index(default) if default in choices else 0
    return st.selectbox(feature, choices, index=index, key=f"{key_prefix}_{feature}")


def prediction_page():
    ensure_project_ready()
    df = cached_data()
    numerical, ordinal, nominal, ordinal_map = cached_schema()

    st.markdown(
        '<div class="hero"><h1>🏠 Ames Housing Price Predictor</h1>'
        '<p class="muted">Estimate a residential property’s sale price '
        "using the project's final Support Vector Regression model.</p></div>",
        unsafe_allow_html=True
    )
    with st.form("prediction_form"):
        values = {}

        groups = [
            ("🏡 Property & Location", ["MSSubClass", "MSZoning", "Neighborhood", "BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd"]),
            ("📐 Size & Rooms", ["LotFrontage", "LotArea", "1stFlrSF", "2ndFlrSF", "GrLivArea", "TotRmsAbvGrd", "BedroomAbvGr", "KitchenAbvGr", "FullBath", "HalfBath"]),
            ("🧱 Basement", ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]),
            ("🚗 Garage & Exterior", ["GarageType", "GarageFinish", "GarageCars", "GarageArea", "GarageYrBlt", "GarageQual", "GarageCond", "ExterQual", "ExterCond", "Fireplaces", "FireplaceQu"]),
            ("🌳 Outdoor & Other", ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "PoolQC", "Fence", "PavedDrive", "CentralAir", "KitchenQual", "SaleCondition"]),
        ]

        for title, features in groups:
            present = [f for f in features if f in df.columns and f != "SalePrice"]
            if not present:
                continue
            st.subheader(title)
            cols = st.columns(3)
            for idx, feature in enumerate(present):
                with cols[idx % 3]:
                    if feature in numerical:
                        values[feature] = numeric_widget(df, feature, "num")
                    elif feature in ordinal:
                        values[feature] = categorical_widget(df, feature, ordinal_map, "ord")
                    else:
                        values[feature] = categorical_widget(df, feature, {}, "cat")

        # Include any features from the actual notebook schema that were not in the curated groups.
        used = set(values)
        remaining = [f for f in numerical + ordinal + nominal if f not in used]
        if remaining:
            with st.expander("Additional model features"):
                cols = st.columns(3)
                for idx, feature in enumerate(remaining):
                    with cols[idx % 3]:
                        if feature in numerical:
                            values[feature] = numeric_widget(df, feature, "extra_num")
                        elif feature in ordinal:
                            values[feature] = categorical_widget(df, feature, ordinal_map, "extra_ord")
                        else:
                            values[feature] = categorical_widget(df, feature, {}, "extra_cat")

        submitted = st.form_submit_button("🔮 Predict Sale Price", use_container_width=True)

    if submitted:
        try:
            # Match the exact feature set expected by the notebook's model.
            feature_order = numerical + ordinal + nominal
            X_input = pd.DataFrame([{feature: values[feature] for feature in feature_order}])
            model = cached_model()
            prediction = float(model.predict(X_input)[0])
            prediction = max(0.0, prediction)

            st.success("Prediction generated successfully.")
            st.markdown(
                f'<div class="hero"><div class="muted">Estimated Sale Price</div><div class="price">${prediction:,.0f}</div><div class="muted">This is a model estimate, not a guaranteed market value.</div></div>',
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def performance_page():
    st.title("📊 Model Performance")
    st.caption("Validation results reported by the existing notebook, plus the final untouched-test evaluation.")

    comparison = pd.DataFrame([
        ["Baseline SVR", 13403.27, 365461700, 19117.05, 0.94],
        ["Tuned SVR", 14149.74, 498378100, 22324.38, 0.91],
        ["Linear Regression", 14324.72, 511951500, 22626.35, 0.91],
        ["Tuned KNN", 17700.29, 770994000, 27766.78, 0.87],
        ["Baseline KNN", 20328.73, 979653300, 31299.41, 0.83],
    ], columns=["Model", "MAE", "MSE", "RMSE", "R²"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected model", "Baseline SVR")
    c2.metric("Validation RMSE", "$19,117")
    c3.metric("Final test R²", "0.91")

    st.subheader("Validation model comparison")
    st.dataframe(comparison.style.format({"MAE": "${:,.0f}", "MSE": "${:,.0f}", "RMSE": "${:,.0f}", "R²": "{:.2f}"}), use_container_width=True, hide_index=True)
    st.bar_chart(comparison.set_index("Model")["RMSE"])

    st.subheader("Final test-set evaluation")
    final = pd.DataFrame([["Train + Validation", 11588.53, 364170900, 19083.26, 0.94], ["Test", 13832.16, 621779900, 24935.52, 0.91]], columns=["Split", "MAE", "MSE", "RMSE", "R²"])
    st.dataframe(final.style.format({"MAE": "${:,.0f}", "MSE": "${:,.0f}", "RMSE": "${:,.0f}", "R²": "{:.2f}"}), use_container_width=True, hide_index=True)


def importance_page():
    st.title("🔎 Feature Importance")
    st.caption("Permutation importance from the notebook's Baseline SVR validation analysis.")
    imp = pd.DataFrame([
        ["GrLivArea",14200.3737,932.2221],["OverallQual",10895.5277,1091.3589],["TotalBsmtSF",4481.3009,608.7946],["LotArea",3414.6871,302.2482],["1stFlrSF",3088.6662,568.2065],["YearBuilt",2363.7889,360.4921],["OverallCond",2181.6896,461.3714],["BsmtQual",2021.3921,326.6919],["FullBath",1673.5227,342.2478],["Neighborhood",1075.5366,138.4619],["GarageArea",1041.6634,245.8551],["YearRemodAdd",891.1874,339.2767],["KitchenQual",814.0212,230.5203],["BsmtExposure",760.6009,264.8627],["GarageCars",730.8009,219.8937],["ExterQual",676.0401,195.4198],["Fireplaces",583.3355,217.0998],["SaleCondition",529.8286,71.7886],["2ndFlrSF",510.1897,216.1993],["FireplaceQu",477.0804,218.4824]
    ], columns=["Feature","Importance","Std"])
    st.bar_chart(imp.set_index("Feature")["Importance"].sort_values())
    st.dataframe(imp.style.format({"Importance":"{:,.1f}","Std":"{:,.1f}"}), use_container_width=True, hide_index=True)
    st.info("Permutation importance measures the decrease in model performance after a feature is shuffled. It indicates model reliance, not causation.")


def model_info_page():
    st.title("🤖 Model Information")
    st.markdown("""
    ### Final model: Support Vector Regression (SVR)

    The notebook compares Linear Regression, baseline SVR, baseline KNN, tuned SVR, and tuned KNN. The **Baseline SVR** is selected because it achieved the best validation RMSE and R².

    **Target:** `SalePrice`

    **Target transformation:** `log1p` during training and `expm1` for predictions.

    **Preprocessing:**
    - Skewed numerical features → median imputation → Yeo-Johnson → StandardScaler
    - Other numerical features → median imputation → StandardScaler
    - Nominal categorical features → most-frequent imputation → OneHotEncoder
    - Ordinal categorical features → most-frequent imputation → OrdinalEncoder using the notebook's explicit category order

    The Streamlit app does not recreate these transformations. It uses the fitted model produced from the existing notebook logic.
    """)


def about_page():
    st.title("ℹ️ About")
    st.markdown("""
    **Ames Housing Price Prediction** is a regression project built around the Ames Housing dataset.

    The original Jupyter notebook is intentionally left **unchanged**. This Streamlit application is a separate interface layer around its final model.

    The goal is to make the ML project interactive without changing the modeling decisions, preprocessing definitions, or notebook narrative.
    """)
    st.code("streamlit run app.py", language="bash")


if page == "Price Prediction":
    prediction_page()
elif page == "Model Performance":
    performance_page()
elif page == "Feature Importance":
    importance_page()
elif page == "Model Information":
    model_info_page()
else:
    about_page()