import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# ────────────────────────────────────────────────────────────────
# Train & save the model if healthcare.pkl doesn't exist yet
# ────────────────────────────────────────────────────────────────
MODEL_PATH = 'healthcare.pkl'

def train_and_save(csv_path='healthcare_cleaned.csv'):
    df = pd.read_csv(csv_path)
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date']    = pd.to_datetime(df['Discharge Date'])
    df['Length of Stay']    = (df['Discharge Date'] - df['Date of Admission']).dt.days

    features = ['Age', 'Gender', 'Blood Type', 'Medical Condition',
                'Insurance Provider', 'Billing Amount', 'Admission Type',
                'Medication', 'Length of Stay']
    X = df[features]
    y = df['Test Results']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = ['Age', 'Billing Amount', 'Length of Stay']
    cat_cols = ['Gender', 'Blood Type', 'Medical Condition',
                'Insurance Provider', 'Admission Type', 'Medication']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore')),
        ('scaler', StandardScaler(with_mean=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    model = Pipeline([
        ('preprocessing', preprocessor),
        ('model', RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42))
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    # Auto-train if pkl not found but CSV is present
    if os.path.exists('healthcare_cleaned.csv'):
        st.info("Training model for the first time, please wait…")
        return train_and_save()
    st.error("healthcare.pkl not found. Please run the notebook first to train the model.")
    return None


# ────────────────────────────────────────────────────────────────
# Prediction helper
# ────────────────────────────────────────────────────────────────
def predict_test_result(classifier, age, gender, blood_type, condition,
                         insurance, billing, admission_type,
                         medication, length_of_stay):
    sample = pd.DataFrame({
        'Age':                [int(age)],
        'Gender':             [gender],
        'Blood Type':         [blood_type],
        'Medical Condition':  [condition],
        'Insurance Provider': [insurance],
        'Billing Amount':     [float(billing)],
        'Admission Type':     [admission_type],
        'Medication':         [medication],
        'Length of Stay':     [int(length_of_stay)]
    })
    return classifier.predict(sample)[0]


# ────────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Healthcare Test Result Predictor",
        page_icon="🏥",
        layout="wide"
    )

    # Header banner (same style as Titanic app)
    st.markdown("""
        <div style="background-color:#1f77b4;padding:14px;border-radius:6px">
        <h2 style="color:white;text-align:center;">🏥 Healthcare Test Result Prediction App</h2>
        <p style="color:#d0e8ff;text-align:center;">
            CDS3533 — Supervised Learning (Classification) — Random Forest
        </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    classifier = load_model()
    if classifier is None:
        return

    # ── Input form ──────────────────────────────────────────────
    st.subheader("📋 Enter Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demographics**")
        age    = st.slider("Age", 0, 100, 35)
        gender = st.radio("Gender", ["Male", "Female", "Other"])
        blood_type = st.selectbox(
            "Blood Type",
            ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
        )

    with col2:
        st.markdown("**🏨 Admission Details**")
        condition = st.selectbox(
            "Medical Condition",
            ["Cancer", "Obesity", "Diabetes", "Asthma",
             "Hypertension", "Arthritis", "Heart Disease",
             "COVID-19", "Flu", "Migraine"]
        )
        admission_type = st.radio(
            "Admission Type",
            ["Urgent", "Emergency", "Elective"]
        )
        length_of_stay = st.slider("Length of Stay (days)", 0, 60, 5)

    with col3:
        st.markdown("**💊 Treatment & Finance**")
        medication = st.selectbox(
            "Medication",
            ["Paracetamol", "Ibuprofen", "Aspirin", "Penicillin",
             "Lipitor", "Metformin", "Albuterol", "Lisinopril", "Atorvastatin"]
        )
        insurance = st.selectbox(
            "Insurance Provider",
            ["Blue Cross", "Medicare", "Aetna",
             "UnitedHealthcare", "Cigna", "Unknown"]
        )
        billing = st.number_input(
            "Billing Amount ($)", 0.0, 40000.0, 10000.0, step=500.0
        )

    st.markdown("---")

    # ── Predict button ───────────────────────────────────────────
    if st.button("🔍 Predict Test Result", use_container_width=True):
        result = predict_test_result(
            classifier, age, gender, blood_type, condition,
            insurance, billing, admission_type, medication, length_of_stay
        )

        color_map = {
            "Normal":       "#28a745",
            "Abnormal":     "#dc3545",
            "Inconclusive": "#fd7e14",
            "Pending":      "#007bff"
        }
        icon_map = {
            "Normal":       "✅",
            "Abnormal":     "⚠️",
            "Inconclusive": "🔶",
            "Pending":      "🔵"
        }
        color = color_map.get(result, "#6c757d")
        icon  = icon_map.get(result, "❓")

        st.markdown(f"""
            <div style="background-color:{color};padding:16px;border-radius:8px;text-align:center">
            <h2 style="color:white;">{icon} Predicted Test Result: <strong>{result}</strong></h2>
            </div>
        """, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
        <p style="text-align:center;color:grey;font-size:12px;">
        CDS3533 Group 2 — Healthcare ML Project — Supervised Classification
        </p>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
