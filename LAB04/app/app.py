"""
Medical Insurance Charges — Statistical Analysis Dashboard
DS602 Lab 4

Three-tab Streamlit app:
1. Data Exploration — interactive filters + reactive plots
2. Hypothesis Testing Lab — dynamic test selection
3. Live Prediction & Diagnostics — OLS predictions + residual plots
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats
import statsmodels.api as sm

st.set_page_config(
    page_title="Medical Insurance — Statistical Dashboard",
    page_icon="🏥",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "insurance.csv"
MODEL_PATH = BASE_DIR / "models" / "insurance_ols.pkl"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model_artifacts():
    return joblib.load(MODEL_PATH)


df = load_data()
artifacts = load_model_artifacts()
model = artifacts["model"]
feature_names = artifacts["feature_names"]
baselines = artifacts["categorical_baselines"]


def build_design_row(age, sex, bmi, children, smoker, region):
    row = {name: 0.0 for name in feature_names}
    row["const"] = 1.0
    row["age"] = float(age)
    row["bmi"] = float(bmi)
    row["children"] = float(children)

    if sex != baselines["sex"]:
        row["sex_" + sex] = 1.0
    if smoker != baselines["smoker"]:
        row["smoker_" + smoker] = 1.0
    if region != baselines["region"]:
        row["region_" + region] = 1.0

    return pd.DataFrame([row])[feature_names]


def predict_with_interval(design_row, alpha=0.05):
    pred = model.get_prediction(design_row)
    summary = pred.summary_frame(alpha=alpha)
    return {
        "mean": float(summary["mean"].iloc[0]),
        "mean_ci_low": float(summary["mean_ci_lower"].iloc[0]),
        "mean_ci_high": float(summary["mean_ci_upper"].iloc[0]),
        "obs_ci_low": float(summary["obs_ci_lower"].iloc[0]),
        "obs_ci_high": float(summary["obs_ci_upper"].iloc[0]),
    }


def run_hypothesis_test(factor, metric):
    groups = sorted(df[factor].dropna().unique().tolist())
    samples = [df.loc[df[factor] == g, metric].dropna().values for g in groups]

    if len(groups) < 2:
        return {"error": "Need at least 2 groups."}

    res = {"factor": factor, "metric": metric, "groups": groups,
           "n": [len(s) for s in samples]}

    shapiro = [stats.shapiro(s) for s in samples if len(s) >= 3]
    normal = all(p > 0.05 for _, p in shapiro)
    levene = stats.levene(*samples)
    equal_var = levene.pvalue > 0.05

    res["shapiro"] = [(round(w, 4), round(p, 6)) for w, p in shapiro]
    res["levene"] = (round(levene.statistic, 4), round(levene.pvalue, 6))
    res["normal"] = normal
    res["equal_var"] = equal_var

    if len(groups) == 2:
        if normal and equal_var:
            s, p = stats.ttest_ind(*samples, equal_var=True)
            res["test"] = "Student's t-test"
        elif normal and not equal_var:
            s, p = stats.ttest_ind(*samples, equal_var=False)
            res["test"] = "Welch's t-test"
        else:
            s, p = stats.mannwhitneyu(*samples, alternative="two-sided")
            res["test"] = "Mann-Whitney U test"
    else:
        if normal and equal_var:
            s, p = stats.f_oneway(*samples)
            res["test"] = "One-Way ANOVA"
        else:
            s, p = stats.kruskal(*samples)
            res["test"] = "Kruskal-Wallis H test"

    res["stat"] = round(float(s), 4)
    res["p"] = float(p)
    res["reject_h0"] = res["p"] < 0.05
    return res


st.title("🏥 Medical Insurance — Statistical Dashboard")
st.caption(
    "DS602 Lab 4 · Hitesh Rathod (202618040) · "
    "OLS regression + hypothesis testing on the Medical Cost dataset"
)

tab1, tab2, tab3 = st.tabs(
    ["📊 Data Exploration", "🧪 Hypothesis Testing Lab", "🔮 Prediction & Diagnostics"]
)

# ============================================================
# TAB 1
# ============================================================
with tab1:
    st.subheader("Interactive Data Explorer")

    with st.sidebar:
        st.header("Filters")

        age_range = st.slider(
            "Age range",
            int(df["age"].min()), int(df["age"].max()),
            (int(df["age"].min()), int(df["age"].max())),
        )

        bmi_range = st.slider(
            "BMI range",
            float(df["bmi"].min()), float(df["bmi"].max()),
            (float(df["bmi"].min()), float(df["bmi"].max())),
            step=0.5,
        )

        regions_all = sorted(df["region"].unique().tolist())
        regions_sel = st.multiselect(
            "Region", options=regions_all, default=regions_all,
        )

        smokers_all = sorted(df["smoker"].unique().tolist())
        smokers_sel = st.multiselect(
            "Smoker status", options=smokers_all, default=smokers_all,
        )

    # Guard against empty selection
    if not regions_sel:
        regions_sel = regions_all
    if not smokers_sel:
        smokers_sel = smokers_all

    mask = (
        df["age"].between(age_range[0], age_range[1])
        & df["bmi"].between(bmi_range[0], bmi_range[1])
        & df["region"].isin(regions_sel)
        & df["smoker"].isin(smokers_sel)
    )
    filtered = df[mask].copy()

    st.write(f"**{len(filtered):,} rows** match the current filters.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean charges", f"${filtered['charges'].mean():,.0f}")
    c2.metric("Median charges", f"${filtered['charges'].median():,.0f}")
    c3.metric("Std dev", f"${filtered['charges'].std():,.0f}")
    c4.metric("IQR", f"${filtered['charges'].quantile(0.75) - filtered['charges'].quantile(0.25):,.0f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Distribution of charges**")
        fig = px.histogram(
            filtered, x="charges", nbins=50,
            color="smoker", marginal="box",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")

    with col_right:
        st.markdown("**Charges by region**")
        fig = px.box(
            filtered, x="region", y="charges", color="region",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(height=400, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**BMI vs Charges (interactive)**")
    fig = px.scatter(
        filtered, x="bmi", y="charges",
        color="smoker", symbol="sex",
        opacity=0.65, hover_data=["age", "region", "children"],
    )
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Correlation matrix (numeric features)**")
    num_cols = ["age", "bmi", "children", "charges"]
    corr = filtered[num_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Show filtered data"):
        st.dataframe(filtered.head(200), width="stretch")

# ============================================================
# TAB 2
# ============================================================
with tab2:
    st.subheader("Hypothesis Testing Lab")
    st.caption(
        "Choose a categorical factor and a numerical metric. "
        "The app automatically checks normality and variance, "
        "then runs the appropriate test at α = 0.05."
    )

    colA, colB = st.columns(2)
    with colA:
        factor_col = st.selectbox("Categorical factor", ["smoker", "sex", "region"], index=0)
    with colB:
        metric_col = st.selectbox("Numerical metric", ["charges", "bmi", "age"], index=0)

    st.markdown("---")

    result = run_hypothesis_test(factor_col, metric_col)

    if "error" in result:
        st.warning(result["error"])
    else:
        st.markdown(f"### Test: `{result['test']}`")

        c1, c2, c3 = st.columns(3)
        c1.metric("Test statistic", f"{result['stat']}")
        c2.metric("p-value", f"{result['p']:.6f}")
        c3.metric("α", "0.05")

        st.markdown(f"**Groups compared:** {', '.join(result['groups'])}")
        st.markdown(f"**Sample sizes:** {result['n']}")

        with st.expander("Assumption checks"):
            st.write("**Shapiro-Wilk (normality per group):**")
            for g, (w, p) in zip(result["groups"], result["shapiro"]):
                st.write(f"- {g}: W = {w}, p = {p}")
            st.write(f"**Levene (equal variance):** stat = {result['levene'][0]}, p = {result['levene'][1]}")
            st.write(f"Normality satisfied: **{result['normal']}**")
            st.write(f"Equal variance: **{result['equal_var']}**")

        if result["reject_h0"]:
            st.error(
                f"**Decision: REJECT H₀** · p = {result['p']:.6f} < 0.05\n\n"
                f"There is a statistically significant difference in **{metric_col}** "
                f"across the groups of **{factor_col}**."
            )
        else:
            st.success(
                f"**Decision: FAIL TO REJECT H₀** · p = {result['p']:.6f} ≥ 0.05\n\n"
                f"No statistically significant difference in **{metric_col}** "
                f"across the groups of **{factor_col}**."
            )

        st.markdown("### Group distributions")
        fig = px.box(
            df, x=factor_col, y=metric_col, color=factor_col,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")

# ============================================================
# TAB 3
# ============================================================
with tab3:
    st.subheader("Live Prediction & Diagnostics")
    st.caption("Enter listing details. The OLS model returns a predicted charge with 95% intervals.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 64, 35)
        bmi = st.slider("BMI", 15.0, 55.0, 28.0, step=0.1)
        children = st.slider("Children", 0, 5, 0)
    with col2:
        sex = st.selectbox("Sex", ["female", "male"], index=0)
        smoker = st.selectbox("Smoker", ["no", "yes"], index=0)
        region = st.selectbox(
            "Region",
            ["northeast", "northwest", "southeast", "southwest"],
            index=0,
        )

    st.markdown("---")

    if st.button("Predict charges", type="primary", width="stretch"):
        design_row = build_design_row(age, sex, bmi, children, smoker, region)
        preds = predict_with_interval(design_row)

        st.markdown("### Prediction")
        cA, cB, cC = st.columns(3)
        cA.metric("Predicted charges", f"${preds['mean']:,.0f}")
        cB.metric("95% CI (mean)", f"${preds['mean_ci_low']:,.0f} – ${preds['mean_ci_high']:,.0f}")
        cC.metric("95% PI (individual)", f"${preds['obs_ci_low']:,.0f} – ${preds['obs_ci_high']:,.0f}")

        with st.expander("Input summary"):
            st.dataframe(design_row.T.rename(columns={0: "Value"}), width="stretch")

    st.markdown("---")
    st.markdown("### Model Diagnostics (training residuals)")

    fitted = model.fittedvalues
    residuals = model.resid

    colL, colR = st.columns(2)
    with colL:
        st.markdown("**Residuals vs Fitted**")
        fig = px.scatter(
            x=fitted, y=residuals, opacity=0.5,
            labels={"x": "Fitted values", "y": "Residuals"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")

    with colR:
        st.markdown("**Q-Q Plot of Residuals**")
        (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Residuals"))
        fig.add_trace(go.Scatter(
            x=osm, y=slope * osm + intercept,
            mode="lines", name="Normal line",
            line=dict(color="red", dash="dash"),
        ))
        fig.update_layout(
            height=400,
            xaxis_title="Theoretical quantiles",
            yaxis_title="Sample quantiles",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    cA, cB, cC, cD = st.columns(4)
    cA.metric("R²", f"{model.rsquared:.3f}")
    cB.metric("Adj. R²", f"{model.rsquared_adj:.3f}")
    cC.metric("F-statistic", f"{model.fvalue:,.1f}")
    cD.metric("AIC", f"{model.aic:,.1f}")

    with st.expander("Full OLS summary"):
        st.text(model.summary().as_text())

st.markdown("---")
st.caption("Built for DS602 Lab 4 · Model: OLS on Medical Cost dataset · Test R² = 0.751")
