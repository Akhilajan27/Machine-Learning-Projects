import streamlit as st
import pickle
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease AI Dashboard",
    page_icon="❤️",
    layout="wide"
)

# ---------------- SAFE CSS (NO INPUT BUGS) ----------------
st.markdown("""
<style>

/* ================= BACKGROUND ================= */
.stApp {
    background: linear-gradient(135deg, #f4f8fb, #e3f2fd);
}

/* ================= TITLE ================= */
h1 {
    text-align: center;
    color: #0D47A1;
    font-size: 42px;
    font-weight: bold;
}

/* ================= SIDEBAR ================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D47A1, #1565C0);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* ================= INPUT FIX (FINAL & SAFE) ================= */

/* Number input */
.stNumberInput input {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Text input */
.stTextInput input {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* BaseWeb input fallback */
div[data-baseweb="input"] input {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Global fallback */
input {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Labels */
label {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* ================= BUTTON ================= */
.stButton > button {
    background: linear-gradient(90deg, #1565C0, #42A5F5);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    border: none;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0px 5px 20px rgba(21,101,192,0.4);
}

/* ================= IMAGE ================= */
img {
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model_knn.save1", "rb"))
scaler = pickle.load(open("scaler_knn.save1", "rb"))

# ---------------- HEADER ----------------
st.markdown("""
<h1>❤️ Heart Disease Prediction System</h1>
<div style='text-align:center; font-size:18px; color:#455A64;'>
AI Powered Healthcare Dashboard using Machine Learning (KNN)
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- IMAGE ----------------
image = Image.open(r"E:\Dataspark_project\HeartDisease_PredictionKNN\WhatsApp Image 2026-06-02 at 3.12.24 PM.jpeg")
st.image(image, use_container_width=True)

st.markdown("---")

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.title("🧾 Patient Details")

age = st.sidebar.number_input("Age", 1, 120, 30)

sex = st.sidebar.radio("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp = st.sidebar.number_input("Chest Pain Type", 0.0)
trestbps = st.sidebar.number_input("Resting Blood Pressure", 0.0)
chol = st.sidebar.number_input("Cholesterol", 0.0)
fbs = st.sidebar.number_input("Fasting Blood Sugar", 0.0)
restecg = st.sidebar.number_input("Rest ECG", 0.0)
thalach = st.sidebar.number_input("Max Heart Rate", 0.0)
exang = st.sidebar.number_input("Exercise Angina", 0.0)
oldpeak = st.sidebar.number_input("Old Peak", 0.0)
slope = st.sidebar.number_input("Slope", 0.0)
ca = st.sidebar.number_input("CA", 0.0)
thal = st.sidebar.number_input("Thal", 0.0)

# ---------------- GAUGE ----------------
def show_gauge(risk):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={'text': "Heart Disease Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- PREDICTION ----------------
def predict():
    features = np.array([[
        float(age),
        int(sex),
        float(cp),
        float(trestbps),
        float(chol),
        float(fbs),
        float(restecg),
        float(thalach),
        float(exang),
        float(oldpeak),
        float(slope),
        float(ca),
        float(thal)
    ]])

    scaled = scaler.transform(features)

    pred = model.predict(scaled)[0]

    try:
        prob = model.predict_proba(scaled)[0][1]
    except:
        prob = 0

    return pred, prob

# ---------------- BUTTON ----------------
if st.button("🔍 Predict Heart Disease Risk"):

    pred, prob = predict()
    risk = int(prob * 100)

    st.markdown("## 🧾 Result Analysis")

    show_gauge(risk)

    if pred == 0:
        st.success("🟢 LOW RISK: No Heart Disease Detected")
        st.balloons()
    else:
        st.error("🔴 HIGH RISK: Heart Disease Detected")

    if risk < 30:
        st.info("🟢 Low Risk Patient")
    elif risk < 70:
        st.warning("🟡 Moderate Risk Patient")
    else:
        st.error("🔴 High Risk Patient")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#607D8B;'>
AI Medical Dashboard | Built with Streamlit ❤️
</div>
""", unsafe_allow_html=True)