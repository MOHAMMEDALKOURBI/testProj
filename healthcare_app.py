import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

@st.cache_resource
def load_or_train_model():
    MODEL_PATH = 'healthcare.pkl'
    CSV_PATH   = 'healthcare_cleaned.csv'

    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    if not os.path.exists(CSV_PATH):
        return None

    df = pd.read_csv(CSV_PATH)
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date']    = pd.to_datetime(df['Discharge Date'])
    df['Length of Stay']    = (df['Discharge Date'] - df['Date of Admission']).dt.days

    X = df[['Age','Gender','Blood Type','Medical Condition',
            'Insurance Provider','Billing Amount','Admission Type',
            'Medication','Length of Stay']]
    y = df['Test Results']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    num_cols = ['Age','Billing Amount','Length of Stay']
    cat_cols = ['Gender','Blood Type','Medical Condition',
                'Insurance Provider','Admission Type','Medication']

    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                         ('sc',  StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                         ('enc', OneHotEncoder(drop='first', handle_unknown='ignore')),
                         ('sc',  StandardScaler(with_mean=False))])
    prep = ColumnTransformer([('num', num_pipe, num_cols),
                              ('cat', cat_pipe, cat_cols)])
    model = Pipeline([('preprocessing', prep),
                      ('model', RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42))])
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model


def predict(classifier, age, gender, blood_type, condition,
            insurance, billing, admission_type, medication, los):
    sample = pd.DataFrame({
        'Age': [int(age)], 'Gender': [gender], 'Blood Type': [blood_type],
        'Medical Condition': [condition], 'Insurance Provider': [insurance],
        'Billing Amount': [float(billing)], 'Admission Type': [admission_type],
        'Medication': [medication], 'Length of Stay': [int(los)]
    })
    return classifier.predict(sample)[0]


def main():
    st.set_page_config(page_title="Healthcare Predictor", page_icon="🏥", layout="wide")

    st.markdown("""
        <div style="background-color:#1f77b4;padding:14px;border-radius:6px">
        <h2 style="color:white;text-align:center;">🏥 Healthcare Test Result Prediction App</h2>
        <p style="color:#d0e8ff;text-align:center;">
            CDS3533 — Supervised Learning (Classification) — Random Forest
        </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("⏳ Loading model... first run may take ~1 minute to train"):
        classifier = load_or_train_model()

    if classifier is None:
        st.error("❌ healthcare_cleaned.csv not found. Please add it to your GitHub repo.")
        return

    st.success("✅ Model ready!")
    st.markdown("---")
    st.subheader("📋 Enter Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demographics**")
        age        = st.slider("Age", 0, 100, 35)
        gender     = st.radio("Gender", ["Male","Female","Other"])
        blood_type = st.selectbox("Blood Type", ["A+","A-","B+","B-","O+","O-","AB+","AB-"])

    with col2:
        st.markdown("**🏨 Admission Details**")
        condition      = st.selectbox("Medical Condition",
                            ["Cancer","Obesity","Diabetes","Asthma","Hypertension",
                             "Arthritis","Heart Disease","COVID-19","Flu","Migraine"])
        admission_type = st.radio("Admission Type", ["Urgent","Emergency","Elective"])
        los            = st.slider("Length of Stay (days)", 0, 60, 5)

    with col3:
        st.markdown("**💊 Treatment & Finance**")
        medication = st.selectbox("Medication",
                        ["Paracetamol","Ibuprofen","Aspirin","Penicillin",
                         "Lipitor","Metformin","Albuterol","Lisinopril","Atorvastatin"])
        insurance  = st.selectbox("Insurance Provider",
                        ["Blue Cross","Medicare","Aetna","UnitedHealthcare","Cigna","Unknown"])
        billing    = st.number_input("Billing Amount ($)", 0.0, 40000.0, 10000.0, step=500.0)

    st.markdown("---")

    if st.button("🔍 Predict Test Result", use_container_width=True):
        result = predict(classifier, age, gender, blood_type, condition,
                         insurance, billing, admission_type, medication, los)
        color = {"Normal":"#28a745","Abnormal":"#dc3545",
                 "Inconclusive":"#fd7e14","Pending":"#007bff"}.get(result,"#6c757d")
        icon  = {"Normal":"✅","Abnormal":"⚠️",
                 "Inconclusive":"🔶","Pending":"🔵"}.get(result,"❓")
        st.markdown(f"""
            <div style="background-color:{color};padding:20px;border-radius:8px;text-align:center">
            <h2 style="color:white;">{icon} Predicted Test Result: <strong>{result}</strong></h2>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="text-align:center;color:grey;font-size:12px;">CDS3533 Group 2 — Healthcare ML Project</p>',
                unsafe_allow_html=True)

if __name__ == '__main__':
    main()
