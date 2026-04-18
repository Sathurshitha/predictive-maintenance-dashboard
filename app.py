from pathlib import Path
import warnings

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="altair")

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "final_multiclass_xgboost_ros_model.pkl"
ENCODER_PATH = APP_DIR / "label_encoder.pkl"
FEATURES_PATH = APP_DIR / "feature_columns.pkl"

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
:root {
    --bg: #07111f;
    --bg2: #0b1730;
    --panel: rgba(15, 23, 42, 0.88);
    --panel-2: rgba(30, 41, 59, 0.92);
    --border: rgba(148, 163, 184, 0.20);
    --text: #e5eefc;
    --muted: #9fb1cc;
    --accent: #3b82f6;
    --accent2: #06b6d4;
    --green: #16a34a;
    --yellow: #d97706;
    --red: #dc2626;
    --shadow: 0 18px 40px rgba(0,0,0,0.28);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(6,182,212,0.12), transparent 24%),
        linear-gradient(180deg, #07101e 0%, #0b1220 100%);
    color: var(--text);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

h1, h2, h3, h4, h5, h6, p, label, div, span {
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081223 0%, #0a1426 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.10);
}

.hero-wrap {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    border: 1px solid rgba(96, 165, 250, 0.28);
    background:
        linear-gradient(135deg, rgba(37,99,235,0.28) 0%, rgba(14,23,42,0.92) 48%, rgba(8,15,30,0.96) 100%);
    box-shadow: var(--shadow);
    margin-bottom: 20px;
}

.hero-glow-1 {
    position: absolute;
    width: 420px;
    height: 420px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(59,130,246,0.28), transparent 65%);
    top: -150px;
    right: -90px;
    pointer-events: none;
}

.hero-glow-2 {
    position: absolute;
    width: 300px;
    height: 300px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(6,182,212,0.18), transparent 65%);
    bottom: -120px;
    left: -60px;
    pointer-events: none;
}

.hero-grid {
    position: relative;
    z-index: 2;
    display: grid;
    grid-template-columns: 1.25fr 0.85fr;
    gap: 20px;
    align-items: center;
    padding: 30px;
}

@media (max-width: 900px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid rgba(148,163,184,0.18);
    background: rgba(255,255,255,0.05);
    color: #dbeafe;
    border-radius: 999px;
    padding: 8px 13px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 14px;
}

.hero-title {
    font-size: 42px;
    line-height: 1.06;
    font-weight: 800;
    margin: 0 0 12px 0;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #ffffff 0%, #dbeafe 35%, #93c5fd 70%, #67e8f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 15px;
    line-height: 1.7;
    color: #d9e6fb;
    max-width: 720px;
    margin-bottom: 18px;
}

.hero-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.hero-pill {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(148,163,184,0.16);
    color: #dbeafe;
    border-radius: 999px;
    padding: 9px 14px;
    font-size: 13px;
    font-weight: 600;
}

.hero-side-card {
    background: linear-gradient(180deg, rgba(15,23,42,0.88), rgba(2,6,23,0.95));
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 22px;
    padding: 18px;
    min-height: 260px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}

.hero-side-title {
    font-size: 16px;
    font-weight: 700;
    color: #dbeafe;
    margin-bottom: 12px;
}

.hero-stat {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 10px;
}

.hero-stat-label {
    color: #94a3b8;
    font-size: 12px;
    margin-bottom: 6px;
}

.hero-stat-value {
    color: #ffffff;
    font-size: 22px;
    font-weight: 800;
}

.section-card {
    background: rgba(17, 24, 39, 0.86);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.16);
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 12px;
    color: #f8fbff;
}

.kpi-card {
    background: rgba(17, 24, 39, 0.90);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px;
    text-align: center;
    min-height: 112px;
}

.kpi-label {
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 28px;
    font-weight: 800;
}

.kpi-sub {
    font-size: 12px;
    color: var(--muted);
    margin-top: 6px;
}

.status-ok {
    background: rgba(22, 163, 74, 0.16);
    border: 1px solid rgba(22, 163, 74, 0.45);
    color: #dcfce7;
    padding: 14px 16px;
    border-radius: 12px;
    font-weight: 600;
    margin-bottom: 14px;
}

.status-warn {
    background: rgba(217, 119, 6, 0.16);
    border: 1px solid rgba(217, 119, 6, 0.45);
    color: #ffedd5;
    padding: 14px 16px;
    border-radius: 12px;
    font-weight: 600;
    margin-bottom: 14px;
}

.status-danger {
    background: rgba(220, 38, 38, 0.16);
    border: 1px solid rgba(220, 38, 38, 0.45);
    color: #fee2e2;
    padding: 14px 16px;
    border-radius: 12px;
    font-weight: 600;
    margin-bottom: 14px;
}

.info-box {
    background: rgba(37, 99, 235, 0.12);
    border: 1px solid rgba(37, 99, 235, 0.35);
    color: #dbeafe;
    padding: 12px 14px;
    border-radius: 12px;
    margin-bottom: 10px;
}

.small-note {
    color: var(--muted);
    font-size: 12px;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    border: 1px solid #2563eb !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    border-radius: 10px !important;
}

hr {
    border-color: rgba(148,163,184,0.18) !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD ARTIFACTS
# =========================================================
@st.cache_resource
def load_artifacts():
    missing = []
    for p in [MODEL_PATH, ENCODER_PATH, FEATURES_PATH]:
        if not p.exists():
            missing.append(str(p))

    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    feat_cols = joblib.load(FEATURES_PATH)
    return model, label_encoder, feat_cols


try:
    model, le, FEAT_COLS = load_artifacts()
    CLASS_NAMES = list(le.classes_)
except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.stop()

# =========================================================
# HELPERS
# =========================================================
BATCH_HELP_COLUMNS = [
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
    "Type",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Type" not in df.columns:
        df["Type"] = "M"

    df["Type"] = df["Type"].astype(str).str.upper().str.strip()
    df["Speed_Torque_Ratio"] = (
        df["Rotational_speed_rpm"].astype(float) /
        (df["Torque_Nm"].astype(float) + 1e-6)
    )
    df["Type_L"] = (df["Type"] == "L").astype(int)
    df["Type_M"] = (df["Type"] == "M").astype(int)
    df["Type_H"] = (df["Type"] == "H").astype(int)

    return df


def sanitize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0
    return df[FEAT_COLS]


def overall_fail_prob(probs: np.ndarray) -> float:
    if "No_Failure" in CLASS_NAMES:
        return 1.0 - float(probs[CLASS_NAMES.index("No_Failure")])
    return float(np.max(probs))


def most_likely_subtype(probs: np.ndarray):
    fail_classes = [c for c in CLASS_NAMES if c != "No_Failure"]
    if not fail_classes:
        idx = int(np.argmax(probs))
        return CLASS_NAMES[idx], float(probs[idx])

    fail_scores = [float(probs[CLASS_NAMES.index(c)]) for c in fail_classes]
    idx = int(np.argmax(fail_scores))
    return fail_classes[idx], fail_scores[idx]


def risk_band(p_fail: float, threshold: float) -> str:
    if p_fail < threshold * 0.5:
        return "Low"
    if p_fail < threshold:
        return "Medium"
    return "High"


def decision_label(p_fail: float, threshold: float) -> str:
    return "Failure Risk" if p_fail >= threshold else "Normal Operation"


def advice_from_features(p_fail: float, sub: str, threshold: float) -> list[str]:
    if p_fail < threshold * 0.5:
        return [
            "Machine is operating in a stable range.",
            "Continue routine inspection and sensor monitoring.",
            "No immediate maintenance action is required."
        ]
    elif p_fail < threshold:
        return [
            "Some parameters are moving away from normal behavior.",
            "Check temperature, torque, and tool wear trends.",
            "Plan preventive maintenance during the next available window."
        ]
    else:
        return [
            f"High failure risk detected. Most likely subtype: {sub}.",
            "Inspect machine condition before continued operation.",
            "Review torque, rotational speed, and temperature settings.",
            "Schedule maintenance or component replacement immediately."
        ]


def build_single_input_df(
    air_temp: float,
    process_temp: float,
    rpm: float,
    torque: float,
    wear: int,
    typ: str
) -> pd.DataFrame:
    raw = pd.DataFrame([{
        "Air_temperature_K": air_temp,
        "Process_temperature_K": process_temp,
        "Rotational_speed_rpm": rpm,
        "Torque_Nm": torque,
        "Tool_wear_min": wear,
        "Type": typ
    }])
    raw = add_engineered_features(raw)
    return sanitize_cols(raw)


def validate_batch_columns(df: pd.DataFrame):
    return [c for c in BATCH_HELP_COLUMNS if c not in df.columns]


def predict_dataframe(df_raw: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df_raw.copy()
    df = add_engineered_features(df)
    X = sanitize_cols(df)

    probs = model.predict_proba(X)

    result = df_raw.copy()
    result["failure_probability"] = [overall_fail_prob(row) for row in probs]
    result["decision"] = result["failure_probability"].apply(lambda x: decision_label(x, threshold))
    result["risk_band"] = result["failure_probability"].apply(lambda x: risk_band(x, threshold))

    top_failure_type = []
    top_failure_prob = []

    for row in probs:
        subtype, subtype_prob = most_likely_subtype(row)
        top_failure_type.append(subtype)
        top_failure_prob.append(subtype_prob)

    result["most_likely_failure_type"] = top_failure_type
    result["most_likely_failure_type_probability"] = top_failure_prob

    for i, cname in enumerate(CLASS_NAMES):
        result[f"prob_{cname}"] = probs[:, i]

    return result

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("Adjust the threshold and choose prediction mode.")

    threshold = st.slider(
        "Failure threshold",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.01,
        help="If failure probability is greater than or equal to this value, the machine is flagged as Failure Risk."
    )

    mode = st.radio(
        "Prediction mode",
        options=["Single Prediction", "Batch Prediction"],
        index=0
    )

# =========================================================
# HEADER
# =========================================================
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-glow-1"></div>
    <div class="hero-glow-2"></div>
    <div class="hero-grid">
        <div>
            <div class="hero-badge">🛠️ Smart Industrial Health Monitoring</div>
            <div class="hero-title">Predictive Maintenance Dashboard</div>
            <div class="hero-subtitle">
                Predict machine failure risk faster with a cleaner workflow for both single-record and batch analysis.
                Tune your threshold, review risk bands, and support maintenance decisions with confidence.
            </div>
            <div class="hero-pill-row">
                <div class="hero-pill">Single Prediction</div>
                <div class="hero-pill">Batch Prediction</div>
                <div class="hero-pill">Threshold: {threshold:.2f}</div>
                <div class="hero-pill">XGBoost Classifier</div>
            </div>
        </div>
        <div class="hero-side-card">
            <div class="hero-side-title">Dashboard Snapshot</div>
            <div class="hero-stat">
                <div class="hero-stat-label">Prediction Mode</div>
                <div class="hero-stat-value">{mode}</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-label">Loaded Classes</div>
                <div class="hero-stat-value">{len(CLASS_NAMES)}</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-label">Default Threshold</div>
                <div class="hero-stat-value">{threshold:.2f}</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Current Threshold</div>
        <div class="kpi-value">{threshold:.2f}</div>
        <div class="kpi-sub">Editable from sidebar</div>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Prediction Mode</div>
        <div class="kpi-value" style="font-size:20px;">{mode}</div>
        <div class="kpi-sub">Choose single or batch workflow</div>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Model Classes</div>
        <div class="kpi-value">{len(CLASS_NAMES)}</div>
        <div class="kpi-sub">{", ".join(CLASS_NAMES[:4])}{'...' if len(CLASS_NAMES) > 4 else ''}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# =========================================================
# SINGLE PREDICTION
# =========================================================
if mode == "Single Prediction":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Single Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Enter one machine record, then click <b>Run Single Prediction</b>.</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.number_input("Air temperature (K)", min_value=200.0, max_value=1200.0, value=298.0, step=0.1)
        process_temp = st.number_input("Process temperature (K)", min_value=200.0, max_value=1200.0, value=310.0, step=0.1)
        wear = st.number_input("Tool wear (min)", min_value=0, max_value=2000, value=100, step=1)

    with col2:
        rpm = st.number_input("Rotational speed (rpm)", min_value=0, max_value=200000, value=1500, step=1)
        torque = st.number_input("Torque (Nm)", min_value=-50.0, max_value=500.0, value=40.0, step=0.1)
        typ = st.selectbox("Machine type", options=["L", "M", "H"], index=1)

    run_single = st.button("Run Single Prediction", type="primary", use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

    if run_single:
        try:
            X1 = build_single_input_df(
                air_temp=air_temp,
                process_temp=process_temp,
                rpm=rpm,
                torque=torque,
                wear=wear,
                typ=typ
            )

            probs = model.predict_proba(X1)[0]
            p_fail = overall_fail_prob(probs)
            subtype, subtype_prob = most_likely_subtype(probs)
            band = risk_band(p_fail, threshold)
            decision = decision_label(p_fail, threshold)

            if p_fail >= threshold:
                st.markdown(
                    f'<div class="status-danger">🚨 {decision} | Failure probability: {p_fail:.1%} | Risk band: {band}</div>',
                    unsafe_allow_html=True
                )
            elif p_fail >= threshold * 0.5:
                st.markdown(
                    f'<div class="status-warn">⚠️ {decision} | Failure probability: {p_fail:.1%} | Risk band: {band}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="status-ok">✅ {decision} | Failure probability: {p_fail:.1%} | Risk band: {band}</div>',
                    unsafe_allow_html=True
                )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Failure Probability", f"{p_fail:.2%}")
            m2.metric("Decision", decision)
            m3.metric("Risk Band", band)
            m4.metric("Top Failure Type", subtype)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Probability Breakdown</div>', unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Probability": probs
            }).sort_values("Probability", ascending=False)

            chart = (
                alt.Chart(prob_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Class:N", sort="-y", title="Class"),
                    y=alt.Y("Probability:Q", title="Probability"),
                    tooltip=[
                        alt.Tooltip("Class:N", title="Class"),
                        alt.Tooltip("Probability:Q", title="Probability", format=".2%")
                    ]
                )
                .properties(height=350)
            )
            st.altair_chart(chart, use_container_width=True)

            display_df = prob_df.copy()
            display_df["Probability"] = display_df["Probability"].map(lambda x: f"{x:.2%}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)
            for i, tip in enumerate(advice_from_features(p_fail, subtype, threshold), start=1):
                st.write(f"{i}. {tip}")
            st.markdown(
                f'<p class="small-note">Most likely failure subtype probability: {subtype_prob:.2%}</p>',
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Single prediction failed: {e}")

# =========================================================
# BATCH PREDICTION
# =========================================================
else:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Upload a CSV file with one row per machine record. The app will return probabilities, decisions, and the most likely failure subtype.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    st.markdown("**Expected CSV columns:**")
    st.code(", ".join(BATCH_HELP_COLUMNS))

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Uploaded Data Preview</div>', unsafe_allow_html=True)
            st.dataframe(raw_df.head(10), use_container_width=True)
            st.markdown(
                f'<p class="small-note">Rows: {len(raw_df)} | Columns: {len(raw_df.columns)}</p>',
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

            missing_cols = validate_batch_columns(raw_df)
            if missing_cols:
                st.error("Your CSV is missing these required columns: " + ", ".join(missing_cols))
                st.stop()

            run_batch = st.button("Run Batch Prediction", type="primary")

            if run_batch:
                results_df = predict_dataframe(raw_df, threshold)

                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Batch Summary</div>', unsafe_allow_html=True)

                total_rows = len(results_df)
                flagged_rows = int((results_df["decision"] == "Failure Risk").sum())
                normal_rows = int((results_df["decision"] == "Normal Operation").sum())
                avg_fail_prob = float(results_df["failure_probability"].mean())

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Records", f"{total_rows}")
                s2.metric("Failure Risk", f"{flagged_rows}")
                s3.metric("Normal Operation", f"{normal_rows}")
                s4.metric("Average Failure Probability", f"{avg_fail_prob:.2%}")

                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
                st.dataframe(results_df, use_container_width=True)

                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results CSV",
                    data=csv_bytes,
                    file_name="predictive_maintenance_results.csv",
                    mime="text/csv"
                )
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Decision Distribution</div>', unsafe_allow_html=True)

                decision_chart_df = results_df["decision"].value_counts().reset_index()
                decision_chart_df.columns = ["Decision", "Count"]

                chart1 = (
                    alt.Chart(decision_chart_df)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("Decision:N", title="Decision"),
                        y=alt.Y("Count:Q", title="Count"),
                        tooltip=["Decision:N", "Count:Q"]
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart1, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Top 20 Highest Failure Probabilities</div>', unsafe_allow_html=True)

                top20 = results_df.sort_values("failure_probability", ascending=False).head(20).reset_index(drop=True)
                top20["record_id"] = top20.index.astype(str)

                chart2 = (
                    alt.Chart(top20)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("record_id:N", title="Record"),
                        y=alt.Y("failure_probability:Q", title="Failure Probability"),
                        color=alt.Color("decision:N", legend=alt.Legend(title="Decision")),
                        tooltip=[
                            alt.Tooltip("record_id:N", title="Record"),
                            alt.Tooltip("failure_probability:Q", title="Failure Probability", format=".2%"),
                            alt.Tooltip("decision:N", title="Decision"),
                            alt.Tooltip("most_likely_failure_type:N", title="Likely Failure Type")
                        ]
                    )
                    .properties(height=360)
                )
                st.altair_chart(chart2, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Predictive Maintenance Dashboard • Supports single and batch prediction • Adjustable threshold • XGBoost model"
)