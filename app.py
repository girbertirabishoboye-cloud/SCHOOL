"""Smart Classroom Energy System - Streamlit dashboard.

An ML model predicts classroom OCCUPANCY from sensor readings (light,
temperature, humidity, CO2, hour). A transparent rule then recommends the
lighting action: ON when occupied AND dark, otherwise OFF.
"""

import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from core import (
    REPORT_DIR, energy_estimate, find_model_path,
    lighting_decision, load_artifact, predict_occupancy,
)

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Smart Classroom Energy System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. PROFESSIONAL EMERALD, CHARCOAL & GOLD THEME
# =========================================================
st.markdown(
    """
    <style>
    :root {
        --background: #141A17;
        --surface: #1D2721;
        --surface-2: #25332A;
        --border: #3A5142;
        --green: #45A36B;
        --green-light: #78D99A;
        --gold: #E7C76A;
        --text: #F3F6F1;
        --muted: #B6C5B9;
    }

    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--background);
        color: var(--text);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.7rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stSidebar"] {
        background: #1A241E;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .stApp h1, .stApp h2, .stApp h3,
    .stApp p, .stApp label {
        color: var(--text);
    }

    .hero {
        background: linear-gradient(120deg, #1E4931 0%, #286B43 58%, #354F35 100%);
        border: 1px solid #548B62;
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }

    .hero h1 {
        color: #FFFFFF;
        font-size: 34px;
        font-weight: 800;
        line-height: 1.2;
        margin: 8px 0 12px 0;
    }

    .hero p {
        color: #E0EEE2;
        font-size: 15px;
        line-height: 1.7;
        margin: 0;
    }

    .eyebrow {
        color: var(--gold);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .section-heading {
        color: var(--gold);
        font-size: 21px;
        font-weight: 750;
        margin: 16px 0 12px 0;
    }

    .small-muted {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
    }

    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 18px 20px;
        min-height: 120px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.12);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 800;
    }

    div[data-testid="stMetricDelta"] {
        color: var(--gold) !important;
    }

    .recommendation {
        border-radius: 16px;
        padding: 23px;
        margin: 8px 0 16px 0;
        background: var(--surface);
    }

    .recommendation-on {
        border: 1px solid #43865A;
        border-left: 6px solid var(--green-light);
        background: #20392A;
    }

    .recommendation-off {
        border: 1px solid #827548;
        border-left: 6px solid var(--gold);
        background: #302F25;
    }

    .recommendation h2 {
        color: #FFFFFF;
        font-size: 25px;
        margin: 0 0 8px 0;
    }

    .recommendation p {
        color: #D5E0D6;
        font-size: 14px;
        line-height: 1.6;
        margin: 0;
    }

    .status-pill {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 30px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 12px;
        background: rgba(255,255,255,0.09);
        color: var(--gold);
    }

    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: #347C50;
        color: #FFFFFF;
        border: 1px solid #68A779;
        border-radius: 10px;
        min-height: 43px;
        font-weight: 700;
        transition: 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: #438F60;
        color: #FFFFFF;
        border-color: var(--gold);
    }

    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput div[data-baseweb="input"],
    .stTextInput div[data-baseweb="input"] {
        background: #25332A;
        border-color: #526B57;
    }

    .stSelectbox div[data-baseweb="select"] *,
    .stNumberInput input,
    .stTextInput input {
        color: #FFFFFF !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        gap: 5px;
        padding: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #D4E0D5;
        border-radius: 8px;
    }

    .stTabs [aria-selected="true"] {
        background: #347C50 !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: var(--border);
    }

    .footer {
        color: #AAB9AD;
        text-align: center;
        font-size: 12px;
        line-height: 1.8;
        padding: 25px 0 8px 0;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }

        .hero {
            padding: 22px;
        }

        .hero h1 {
            font-size: 26px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD THE TRAINED MODEL
# =========================================================
MODEL_PATH = find_model_path()

if MODEL_PATH is None:
    st.error(
        "No trained model (.joblib file) was found. Put "
        "'smart_classroom_final_model.joblib' inside the 'models' folder "
        "(next to app.py), or create it by running train_model.py."
    )
    st.code(
        "Smart-Classroom-Energy-System/\n"
        "├── app.py\n"
        "├── core.py\n"
        "├── requirements.txt\n"
        "└── models/\n"
        "    └── smart_classroom_final_model.joblib"
    )
    st.stop()


@st.cache_resource
def get_artifact(model_path, modified_time):
    return load_artifact(model_path)


try:
    artifact = get_artifact(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(
        "The model file exists but could not be loaded. Re-create it with "
        "train_model.py using the scikit-learn version in requirements.txt."
    )
    st.caption(f"Technical detail: {exc}")
    st.stop()

model_features = artifact["features"]


def load_metrics():
    path = REPORT_DIR / "metrics.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


metrics = load_metrics()


# =========================================================
# 4. HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">SMART CAMPUS • INTELLIGENT ENERGY</div>
        <h1>Smart Classroom Energy System</h1>
        <p>
            A machine-learning model estimates whether a classroom is
            occupied from sensor readings (light, temperature, humidity,
            CO&#8322; and time of day). A transparent rule then recommends
            whether the lights should be ON or OFF, and the dashboard
            estimates the related energy cost.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        '<p class="small-muted">DECISION SUPPORT • MACHINE LEARNING • ENERGY AWARENESS</p>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        f'<p style="text-align:right;color:#B6C5B9;font-size:12px;">'
        f'Last page load: {datetime.now().strftime("%d %b %Y, %H:%M")}'
        f'</p>',
        unsafe_allow_html=True,
    )


# =========================================================
# 5. SIDEBAR INPUTS
# =========================================================
st.sidebar.markdown("## ⚙️ Control Centre")
st.sidebar.caption("Enter the sensor readings, then run a prediction.")
st.sidebar.divider()

with st.sidebar.form("classroom_input_form"):
    st.markdown("### Sensor readings")

    light_level = st.slider(
        "Light level (sensor units)",
        min_value=0.0, max_value=1000.0, value=20.0, step=1.0,
        help="Same units as the training dataset's light column.",
    )
    temperature = st.slider(
        "Temperature (°C)",
        min_value=10.0, max_value=40.0, value=23.5, step=0.1,
    )
    humidity = st.slider(
        "Humidity (%)",
        min_value=10.0, max_value=95.0, value=49.0, step=0.5,
    )
    co2 = st.slider(
        "CO₂ (ppm)",
        min_value=300, max_value=2500, value=600, step=5,
        help="CO₂ rises when people are present in a closed room.",
    )
    hour = st.slider(
        "Hour of the day",
        min_value=0, max_value=23, value=10,
        help="24-hour clock: 10 means 10:00 AM.",
    )

    st.markdown("### Lighting rule")
    dark_threshold = st.number_input(
        "Darkness threshold (light < this means 'dark')",
        min_value=0.0, max_value=1000.0,
        value=float(artifact["dark_threshold"]), step=5.0,
        help="Lights are recommended ON only if the room is occupied AND "
             "the light level is below this value.",
    )

    st.markdown("### Energy estimate settings")
    bulb_power = st.number_input(
        "Lighting power (watts)",
        min_value=1, max_value=5000, value=100, step=10,
    )
    hours_per_day = st.slider(
        "Potential lighting hours per day", min_value=1, max_value=16, value=8,
    )
    days_per_month = st.slider(
        "Operating days per month", min_value=1, max_value=31, value=22,
    )
    electricity_rate = st.number_input(
        "Electricity rate (RWF per kWh)",
        min_value=0.0, max_value=10000.0, value=300.0, step=25.0,
        help="Replace the example with your applicable tariff.",
    )

    submitted = st.form_submit_button("⚡ Run Prediction",
                                      use_container_width=True)

st.sidebar.divider()
st.sidebar.markdown("### Model information")
st.sidebar.caption(f"Algorithm: {artifact['model_name']}")
st.sidebar.caption(f"Input features: {', '.join(model_features)}")
st.sidebar.caption(f"Model file: {MODEL_PATH.name}")


# =========================================================
# 6. PREDICTION
# =========================================================
available_values = {
    "light_level": light_level,
    "temperature": temperature,
    "humidity": humidity,
    "co2": co2,
    "hour": hour,
}

try:
    occupancy_probability, occupied = predict_occupancy(
        artifact, available_values)
except Exception as exc:
    st.error(
        "Prediction failed. The model's feature names may not match the "
        "dashboard inputs."
    )
    st.code(f"Model features: {model_features}")
    st.caption(f"Technical detail: {exc}")
    st.stop()

prediction, explanation = lighting_decision(
    occupied, light_level, dark_threshold)
occupancy_label = "Occupied" if occupied else "Empty"

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if submitted:
    st.session_state.prediction_history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Light": light_level,
        "Temperature (°C)": temperature,
        "Humidity (%)": humidity,
        "CO2 (ppm)": co2,
        "Hour": hour,
        "Occupancy probability": round(occupancy_probability, 3),
        "Occupancy": occupancy_label,
        "Dark threshold": dark_threshold,
        "Prediction": prediction,
        "Lighting power (W)": bulb_power,
    })


# =========================================================
# 7. MAIN RESULT
# =========================================================
st.markdown(
    '<div class="section-heading">01 / Live classroom recommendation</div>',
    unsafe_allow_html=True,
)

if prediction == "ON":
    st.markdown(
        f"""
        <div class="recommendation recommendation-on">
            <div class="status-pill">LIGHTING RECOMMENDED</div>
            <h2>💡 Lights should be ON</h2>
            <p>{explanation} Review the readings and confirm before acting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="recommendation recommendation-off">
            <div class="status-pill">ENERGY-SAVING OPPORTUNITY</div>
            <h2>🌙 Lights should be OFF</h2>
            <p>{explanation} Verify safety requirements before switching off.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not submitted:
    st.info(
        "Showing the result for the default readings. Adjust the controls "
        "in the sidebar and select **Run Prediction** to record a scenario."
    )


# =========================================================
# 8. KPI CARDS
# =========================================================
daily_energy_kwh, monthly_energy_kwh, monthly_cost = energy_estimate(
    bulb_power, hours_per_day, days_per_month, electricity_rate)

potential_energy_kwh = monthly_energy_kwh if prediction == "OFF" else 0
potential_cost = monthly_cost if prediction == "OFF" else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Lighting recommendation", prediction)
with kpi2:
    st.metric("Predicted occupancy", occupancy_label,
              help="Occupied if the probability reaches the model's "
                   f"decision threshold ({artifact['threshold']:.2f}).")
with kpi3:
    st.metric("Occupancy probability", f"{occupancy_probability:.0%}")
with kpi4:
    st.metric("Light level", f"{light_level:g}")


# =========================================================
# 9. TABS
# =========================================================
tab_analytics, tab_history, tab_eval, tab_about = st.tabs([
    "📊 Energy Analytics",
    "🗂️ Prediction History",
    "✅ Model Evaluation",
    "🧠 Model & Project",
])


# ---------------- TAB 1: ENERGY ANALYTICS ----------------
with tab_analytics:
    st.markdown(
        '<div class="section-heading">Energy use and cost estimates</div>',
        unsafe_allow_html=True,
    )

    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("Daily energy if ON", f"{daily_energy_kwh:.2f} kWh")
    with a2:
        st.metric("Monthly energy if ON", f"{monthly_energy_kwh:.2f} kWh")
    with a3:
        st.metric("Estimated monthly cost if ON", f"{monthly_cost:,.0f} RWF")

    st.markdown(
        """
        <div class="panel">
            <strong>How to interpret these estimates</strong>
            <p class="small-muted">
                These figures assume the configured lighting load runs for
                the selected hours and days. When the recommendation is OFF,
                the displayed value is a hypothetical upper estimate of the
                energy avoided if the load would otherwise have run for the
                whole period. It is not a measured saving.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    energy_data = pd.DataFrame({
        "Scenario": ["Lights ON for configured hours",
                     "Hypothetical energy avoided if OFF"],
        "Energy (kWh)": [monthly_energy_kwh, potential_energy_kwh],
    })

    fig_energy = px.bar(
        energy_data, x="Scenario", y="Energy (kWh)", color="Scenario",
        text_auto=".2f", title="Monthly energy scenario comparison",
        color_discrete_sequence=["#45A36B", "#E7C76A"],
    )
    fig_energy.update_layout(
        paper_bgcolor="#1D2721", plot_bgcolor="#1D2721",
        font_color="#F3F6F1", title_font_color="#E7C76A",
        showlegend=False, margin=dict(l=20, r=20, t=60, b=20),
    )
    fig_energy.update_xaxes(gridcolor="#35483A")
    fig_energy.update_yaxes(gridcolor="#35483A")
    st.plotly_chart(fig_energy, use_container_width=True)

    st.caption(
        f"At the entered tariff the configured load would cost about "
        f"{monthly_cost:,.0f} RWF per month if it ran for all configured "
        f"hours and days."
    )


# ---------------- TAB 2: PREDICTION HISTORY ----------------
with tab_history:
    st.markdown(
        '<div class="section-heading">Prediction records</div>',
        unsafe_allow_html=True,
    )

    history = st.session_state.prediction_history

    if history:
        history_df = pd.DataFrame(history)

        h1, h2 = st.columns(2)
        with h1:
            st.metric("Predictions recorded", len(history_df))
        with h2:
            st.metric("OFF recommendations",
                      int((history_df["Prediction"] == "OFF").sum()))

        st.dataframe(history_df.iloc[::-1], use_container_width=True,
                     hide_index=True)

        st.download_button(
            label="⬇️ Download history as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="smart_classroom_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        prediction_counts = (
            history_df["Prediction"].value_counts()
            .rename_axis("Recommendation").reset_index(name="Count")
        )
        fig_history = px.pie(
            prediction_counts, names="Recommendation", values="Count",
            title="Recorded ON and OFF recommendations",
            color="Recommendation",
            color_discrete_map={"ON": "#45A36B", "OFF": "#E7C76A"},
            hole=0.55,
        )
        fig_history.update_layout(
            paper_bgcolor="#1D2721", plot_bgcolor="#1D2721",
            font_color="#F3F6F1", title_font_color="#E7C76A",
        )
        st.plotly_chart(fig_history, use_container_width=True)

        if st.button("Clear session history", type="secondary"):
            st.session_state.prediction_history = []
            st.rerun()

        st.caption("History lives in the current Streamlit session only.")
    else:
        st.info("No prediction history yet. Use the sidebar and click "
                "'Run Prediction' to create your first record.")


# ---------------- TAB 3: MODEL EVALUATION ----------------
def show_evaluation(block, title):
    """Render accuracy, confusion matrix and per-class precision/recall."""
    st.markdown(f"#### {title}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{block['accuracy']:.1%}")
    c2.metric("Majority-class baseline",
              f"{block['majority_class_baseline_accuracy']:.1%}",
              help="Accuracy of always predicting the most common class.")
    c3.metric("Macro F1", f"{block['macro_f1']:.3f}")

    names = block["class_names"]
    cm = pd.DataFrame(block["confusion_matrix"],
                      index=[f"Actual {n}" for n in names],
                      columns=[f"Predicted {n}" for n in names])
    fig = px.imshow(cm, text_auto=True, aspect="auto",
                    color_continuous_scale=["#1D2721", "#78D99A"])
    fig.update_layout(
        paper_bgcolor="#1D2721", plot_bgcolor="#1D2721",
        font_color="#F3F6F1", coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=20, b=20), height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

    per_class = pd.DataFrame(block["per_class"]).T
    per_class.index.name = "Class"
    st.dataframe(
        per_class.rename(columns={"precision": "Precision", "recall": "Recall",
                                  "f1": "F1-score", "support": "Support"}),
        use_container_width=True,
    )


with tab_eval:
    st.markdown(
        '<div class="section-heading">Evaluation on the held-out test set</div>',
        unsafe_allow_html=True,
    )

    if metrics is None:
        st.warning(
            "No evaluation report found yet. Run "
            "`python train_model.py --data data/<your_dataset>.csv` to train "
            "the model on the dataset and create reports/metrics.json. "
            "This tab then shows the confusion matrix, precision and recall "
            "for both classes."
        )
    else:
        ds = metrics["dataset"]
        sp = metrics["split"]
        st.markdown(
            f"""
            <div class="panel">
                <strong>Dataset</strong>
                <p class="small-muted">
                    {ds['citation']}<br>
                    File: {ds['file']} • {ds['rows_used']:,} rows
                    (empty: {ds['class_counts']['empty']:,},
                    occupied: {ds['class_counts']['occupied']:,})<br>
                    Split: {sp['train_rows']:,} train / {sp['test_rows']:,} test
                    (stratified, random_state={sp['random_state']})
                </p>
                <strong>How the labels were created</strong>
                <p class="small-muted">{metrics['label_rule']}</p>
                <strong>Selected model</strong>
                <p class="small-muted">{metrics['model_name']} (decision
                threshold {metrics['decision_threshold']:.2f}, chosen with
                cross-validation on the training data only)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns(2)
        with left:
            show_evaluation(metrics["occupancy_model"],
                            "Occupancy model (what the ML predicts)")
        with right:
            show_evaluation(metrics["lighting_decision"],
                            "Lighting decision ON/OFF (occupancy + rule)")

        st.caption(
            "Always compare accuracy with the majority-class baseline: when "
            "one class dominates, high accuracy alone can be misleading, so "
            "check precision and recall for both classes."
        )


# ---------------- TAB 4: MODEL AND PROJECT ----------------
with tab_about:
    st.markdown(
        '<div class="section-heading">About this system</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel">
            <h3>Project objective</h3>
            <p class="small-muted">
                Show how machine learning on classroom sensor data can support
                lighting decisions and reduce wasted energy. The dashboard
                gives a recommendation; it does not control any equipment.
            </p>
        </div>

        <div class="panel">
            <h3>System workflow</h3>
            <p class="small-muted">
                1. Enter sensor readings (light, temperature, humidity, CO₂, hour).<br>
                2. The trained model estimates the probability the room is occupied.<br>
                3. Occupied if the probability reaches the tuned threshold.<br>
                4. Lights ON if occupied AND light is below the darkness threshold; otherwise OFF.<br>
                5. Review energy estimates and export the recorded predictions.
            </p>
        </div>

        <div class="panel">
            <h3>Current limitations</h3>
            <p class="small-muted">
                Readings are entered manually; no sensor, relay or lighting
                circuit is connected. The darkness threshold is a design
                choice that must be calibrated to the sensor. Performance is
                measured on one dataset and may differ in another building.
                Recommendations require human verification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Loaded model details")
    d1, d2 = st.columns(2)
    with d1:
        st.write("**Model file**")
        st.code(MODEL_PATH.name)
        st.write("**Input features**")
        st.code(", ".join(model_features))
    with d2:
        st.write("**Algorithm**")
        st.code(artifact["model_name"])
        st.write("**Current input row**")
        st.dataframe(
            pd.DataFrame([available_values])[model_features],
            use_container_width=True, hide_index=True,
        )

    fi = REPORT_DIR / "feature_importance.png"
    if fi.is_file():
        st.image(str(fi), caption="Feature influence on occupancy prediction")


# =========================================================
# 10. FOOTER
# =========================================================
st.divider()
st.markdown(
    """
    <div class="footer">
        <strong>SMART CLASSROOM ENERGY SYSTEM</strong><br>
        Machine Learning • Classroom Energy Awareness • Decision Support<br>
        Academic prototype — recommendations require human verification.
    </div>
    """,
    unsafe_allow_html=True,
)
