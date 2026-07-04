#  Student Performance Prediction System

##  Project Overview

This project is a Machine Learning Prediction System developed using Python and Streamlit. The application predicts a student's exam score based on various academic, personal, and environmental factors.

The project demonstrates the complete machine learning workflow, including data preprocessing, model training, evaluation, model selection, and deployment through a web application.

---

## Objective

The objective of this project is to build a machine learning model that can accurately predict student exam scores using historical student performance data.

---

##  Dataset Description

The dataset contains information about **6,607 students** with **19 input features** and **1 target variable**.

### Features include:

- Hours Studied
- Attendance
- Parental Involvement
- Access to Resources
- Extracurricular Activities
- Sleep Hours
- Previous Scores
- Motivation Level
- Internet Access
- Tutoring Sessions
- Family Income
- Teacher Quality
- School Type
- Peer Influence
- Physical Activity
- Learning Disabilities
- Parental Education Level
- Distance from Home
- Gender

### Target Variable

- Exam Score

---

##  Data Preprocessing

The following preprocessing steps were performed before training the models:

- Removed duplicate records
- Handled missing values using the mode for categorical columns
- Encoded categorical variables using Label Encoding
- Split the dataset into training and testing sets
- Scaled numerical features using StandardScaler

---

##  Machine Learning Models Used

Two regression models were trained and compared:

1. Linear Regression
2. Decision Tree Regressor

After evaluation, **Linear Regression** was selected as the final model because it achieved the lowest prediction error.

---

##  Model Evaluation

### Linear Regression

- Mean Absolute Error (MAE): **1.02**
- Mean Squared Error (MSE): **4.40**

### Decision Tree Regressor

- Mean Absolute Error (MAE): **1.73**
- Mean Squared Error (MSE): **10.88**

### Best Model

 Linear Regression

---

## Streamlit Application Features

The application includes:

- Project Overview
- Dataset Information
- Student Input Form
- Exam Score Prediction
- Model Performance Metrics
- Data Visualization
- Insights

---

## Project Structure

```
week3-ml-prediction-system/
│
├── app.py
├── dataset.csv
├── requirements.txt
├── README.md
│
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── model_training.ipynb
│
├── assets/
│   └── screenshots/
│
└── src/
```

---

## Installation

Clone the repository:

```bash
git clone <your-github-repository-link>
```

Move into the project folder:

```bash
cd Week3-Machine-Learning
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

```bash
streamlit run app.py
```

---

## Screenshots

Screenshots of the application are available in:

```
assets/screenshots/
```

These include:

- Home Page
- Student Input Form
- Prediction Result
- Visualizations and Insights

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib

---

## Learning Outcomes

In this project I learned how to:

- Clean datasets
- Train machine learning models
- Evaluate regression models
- Compare model performance
- Save trained models using Joblib
- Build Streamlit application

---

## Author

**Rameen Hussain**

Week 3 Machine Learning Internship Project