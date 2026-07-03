import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Scaler
# -----------------------------
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -----------------------------
# Page Title
# -----------------------------
st.set_page_config(page_title="Student Performance Prediction")

st.title("🎓 Student Performance Prediction System")

# -----------------------------
# Project Overview
# -----------------------------
st.header("📖 Project Overview")

st.write("""
This application predicts a student's exam score using a Machine Learning model trained on student performance data.
""")

# -----------------------------
# Dataset Information
# -----------------------------
st.header("📊 Dataset Information")

st.write("""
- **Dataset Size:** 6607 Students
- **Features:** 19
- **Target Variable:** Exam Score
- **Model Used:** Linear Regression
""")

# -----------------------------
# Input Section
# -----------------------------
st.header("📝 Enter Student Details")

# Numerical Features
hours = st.number_input("Hours Studied", min_value=1, max_value=44, value=20)
attendance = st.number_input("Attendance (%)", min_value=60, max_value=100, value=80)
sleep = st.number_input("Sleep Hours", min_value=4, max_value=10, value=7)
previous = st.number_input("Previous Scores", min_value=50, max_value=100, value=75)
tutoring = st.number_input("Tutoring Sessions", min_value=0, max_value=8, value=1)
physical = st.number_input("Physical Activity", min_value=0, max_value=6, value=3)

# Categorical Features
parental = {"Low":0,"Medium":1,"High":2}[st.selectbox("Parental Involvement",["Low","Medium","High"])]

resources = {"Low":0,"Medium":1,"High":2}[st.selectbox("Access to Resources",["Low","Medium","High"])]

activities = {"No":0,"Yes":1}[st.selectbox("Extracurricular Activities",["No","Yes"])]

motivation = {"Low":0,"Medium":1,"High":2}[st.selectbox("Motivation Level",["Low","Medium","High"])]

internet = {"No":0,"Yes":1}[st.selectbox("Internet Access",["No","Yes"])]

income = {"Low":0,"Medium":1,"High":2}[st.selectbox("Family Income",["Low","Medium","High"])]

teacher = {"Low":0,"Medium":1,"High":2}[st.selectbox("Teacher Quality",["Low","Medium","High"])]

school = {"Public":0,"Private":1}[st.selectbox("School Type",["Public","Private"])]

peer = {"Negative":0,"Neutral":1,"Positive":2}[st.selectbox("Peer Influence",["Negative","Neutral","Positive"])]

learning = {"No":0,"Yes":1}[st.selectbox("Learning Disabilities",["No","Yes"])]

education = {"High School":0,"College":1,"Postgraduate":2}[st.selectbox("Parental Education Level",["High School","College","Postgraduate"])]

distance = {"Near":0,"Moderate":1,"Far":2}[st.selectbox("Distance from Home",["Near","Moderate","Far"])]

gender = {"Female":0,"Male":1}[st.selectbox("Gender",["Female","Male"])]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Exam Score"):

    input_data = pd.DataFrame([[

        hours,
        attendance,
        parental,
        resources,
        activities,
        sleep,
        previous,
        motivation,
        internet,
        tutoring,
        income,
        teacher,
        school,
        peer,
        physical,
        learning,
        education,
        distance,
        gender

    ]], columns=[

        "Hours_Studied",
        "Attendance",
        "Parental_Involvement",
        "Access_to_Resources",
        "Extracurricular_Activities",
        "Sleep_Hours",
        "Previous_Scores",
        "Motivation_Level",
        "Internet_Access",
        "Tutoring_Sessions",
        "Family_Income",
        "Teacher_Quality",
        "School_Type",
        "Peer_Influence",
        "Physical_Activity",
        "Learning_Disabilities",
        "Parental_Education_Level",
        "Distance_from_Home",
        "Gender"

    ])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(f"🎯 Predicted Exam Score: {prediction[0]:.2f}")

# -----------------------------
# Model Performance
# -----------------------------
st.header("📈 Model Performance")

st.write("**Linear Regression**")

st.metric("MAE", "1.02")
st.metric("MSE", "4.40")

st.header("📊 Visualizations")

df = pd.read_csv("dataset.csv")

st.bar_chart(df["Exam_Score"].value_counts().sort_index())

st.write("""
Most students scored between 65 and 70, indicating that the dataset is concentrated around average academic performance.
""")