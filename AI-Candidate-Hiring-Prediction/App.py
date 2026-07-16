import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import base64

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Candidate Shortlisting AI", layout="wide")

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

if "records" not in st.session_state:
    st.session_state.records = []

# ---------------- BACKGROUND ----------------

def add_bg(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>

        /* Main App */
        .stApp {{
            background: linear-gradient(
                        rgba(255,255,255,0.72),
                        rgba(255,255,255,0.72)
                    ),
                    url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# 🔥 CALL BACKGROUND (FIX YOUR PATH)
add_bg(r"E:\Dataspark_project\Projects\Candidate Shortlisting Prediction Using Machine Learning_Final\image.png")

# ---------------- TITLE ----------------
st.markdown("""
<h1 style='margin-bottom:5px;'>
💼 Candidate Shortlisting Prediction System
</h1>

<h4 style='color:#4B6584; margin-top:0;'>
AI Powered HR Recruitment Dashboard
</h4>
""",
unsafe_allow_html=True)
# ---------------- FEATURES ----------------
FEATURES = {
    "Age": (18, 60, 30),
    "Gender": (0, 1, 0),
    "EducationLevel": (1, 4, 2),
    "ExperienceYears": (0, 40, 5),
    "PreviousCompanies": (0, 10, 2),
    "DistanceFromCompany": (0.0, 100.0, 10.0),
    "InterviewScore": (0, 100, 50),
    "SkillScore": (0, 100, 50),
    "PersonalityScore": (0, 100, 50),
    "RecruitmentStrategy": (1, 3, 1)
}

# ---------------- SIDEBAR ----------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "🎯 Prediction", "📊 Dashboard", "📥 Download", "📈 Model Insights"]
)

# ---------------- HOME ----------------
if menu == "🏠 Home":
    st.subheader("Welcome 👋")
    st.write("This AI predicts whether a candidate will be shortlisted based on HR features.")
    st.success("Use the sidebar to navigate.")

# ---------------- PREDICTION ----------------
# ---------------- PREDICTION ----------------
elif menu == "🎯 Prediction":

    st.markdown("""
    <style>

    .glass{
        background: rgba(255,255,255,0.85);
        padding:30px;
        border-radius:20px;
        box-shadow:0 10px 30px rgba(0,0,0,0.15);
        margin-top:20px;
        margin-bottom:20px;
    }

    .title{
        text-align:center;
        font-size:34px;
        font-weight:bold;
        color:#12355B;
        margin-bottom:5px;
    }

    .subtitle{
        text-align:center;
        color:#555;
        margin-bottom:25px;
        font-size:18px;
    }

    .result{
        background:#F8FAFC;
        border-left:8px solid #2563EB;
        border-radius:15px;
        padding:20px;
        margin-top:25px;
        box-shadow:0 5px 15px rgba(0,0,0,.1);
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
    <div class="title">
    👤 Candidate Information
    </div>

    <div class="subtitle">
    Enter candidate details below to predict hiring eligibility.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ---------------- LEFT COLUMN ----------------

    with col1:

        age = st.slider("Age", 18, 60, 30)

        gender = st.selectbox(
            "Gender",
            [0,1],
            format_func=lambda x: "Male" if x==1 else "Female"
        )

        education = st.selectbox(
            "Education Level",
            [1,2,3,4],
            format_func=lambda x:{
                1:"High School",
                2:"Bachelor",
                3:"Master",
                4:"PhD"
            }[x]
        )

        experience = st.slider(
            "Experience (Years)",
            0,
            40,
            5
        )

        previous = st.slider(
            "Previous Companies",
            0,
            10,
            2
        )

    # ---------------- RIGHT COLUMN ----------------

    with col2:

        distance = st.slider(
            "Distance From Company (km)",
            0.0,
            100.0,
            10.0
        )

        interview = st.slider(
            "Interview Score",
            0,
            100,
            70
        )

        skill = st.slider(
            "Skill Score",
            0,
            100,
            75
        )

        personality = st.slider(
            "Personality Score",
            0,
            100,
            70
        )

        strategy = st.selectbox(
            "Recruitment Strategy",
            [1,2,3],
            format_func=lambda x:{
                1:"Internal",
                2:"Referral",
                3:"External"
            }[x]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict = st.button("🚀 Predict Candidate")

    st.markdown("</div>", unsafe_allow_html=True)

    if predict:

        user_data = [[
            age,
            gender,
            education,
            experience,
            previous,
            distance,
            interview,
            skill,
            personality,
            strategy
        ]]

        scaled = scaler.transform(user_data)

        prediction = model.predict(scaled)[0]

        probability = model.predict_proba(scaled)[0][1]

        result = "Selected" if prediction==1 else "Rejected"

        st.session_state.records.append({

            "Age":age,
            "Gender":"Male" if gender==1 else "Female",
            "Education":education,
            "Experience":experience,
            "Prediction":result,
            "Probability":round(probability,2)

        })

        if prediction==1:

            color="#16A34A"
            icon="✅"
            status="LIKELY TO BE HIRED"

        else:

            color="#DC2626"
            icon="❌"
            status="NOT LIKELY TO BE HIRED"

        st.markdown(f"""
        <div class="result">

        <h2 style="color:{color};margin-bottom:10px;">
        {icon} Prediction Result
        </h2>

        <h3 style="color:#12355B;">
        Candidate is <b>{status}</b>
        </h3>

        <hr>

        <p style="font-size:18px;">
        <b>Hiring Probability :</b>
        <span style="color:{color};font-size:24px;font-weight:bold;">
        {probability:.2%}
        </span>
        </p>

        </div>
        """, unsafe_allow_html=True)
# ---------------- DASHBOARD ----------------
elif menu == "📊 Dashboard":
    st.subheader("HR Dashboard")

    if len(st.session_state.records) == 0:
        st.warning("No data yet. Please run predictions first.")
    else:
        df = pd.DataFrame(st.session_state.records)

        st.dataframe(df)

        total = len(df)
        selected = (df["Prediction"] == "Selected").sum()
        rejected = (df["Prediction"] == "Rejected").sum()
        rate = (selected / total) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", total)
        col2.metric("Selected", selected)
        col3.metric("Selection Rate", f"{rate:.2f}%")

        fig, ax = plt.subplots()
        df["Prediction"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

# ---------------- MODEL INSIGHTS ----------------
elif menu == "📈 Model Insights":
    st.subheader("Feature Importance")

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

        df_imp = pd.DataFrame({
            "Feature": list(FEATURES.keys()),
            "Importance": importance
        }).sort_values("Importance")

        fig, ax = plt.subplots()
        ax.barh(df_imp["Feature"], df_imp["Importance"])
        st.pyplot(fig)
    else:
        st.warning("Model does not support feature importance")

# ---------------- DOWNLOAD ----------------
elif menu == "📥 Download":
    st.subheader("Download Sample Output")

    sample = pd.DataFrame({
        "Age": [30],
        "Gender": [1],
        "EducationLevel": [2],
        "ExperienceYears": [5],
        "PreviousCompanies": [2],
        "DistanceFromCompany": [10],
        "InterviewScore": [70],
        "SkillScore": [80],
        "PersonalityScore": [75],
        "RecruitmentStrategy": [2]
    })

    pred = model.predict(scaler.transform(sample))
    sample["Prediction"] = np.where(pred == 1, "Selected", "Rejected")

    st.write(sample)

    csv = sample.to_csv(index=False).encode("utf-8")

    st.download_button("Download CSV", csv, "predictions.csv", "text/csv")