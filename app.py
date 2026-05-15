import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Logistic_Reg_heart_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")


# creating UI

st.title("❤️Heart Stroke Prediction by M")
st.markdown("Provide the following details")

# getting input as original df
age = st.slider("Age", 18,100,40)
sex = st.selectbox("SEX", ['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mm/dL)", 100,600,200)
fastingbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


# in data from which we are predicting has more(16) than this columns, so we will put 1 in input(eg;"Up") and will put 0 in rest(eg;"Flat", "Down")
if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fastingbs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])        # creating input dataframe for model to predict

    for col in expected_columns:                # filling missing values in columns
        if col not in input_df.columns: 
            input_df[col] = 0

    input_df = input_df[expected_columns]       # adding all columns in input_df

    scaled_input = scaler.transform(input_df)   # scaling input_df

    prediction = model.predict(scaled_input)[0] # predicting output using model in pickle file

    # showing result
    if prediction == 1:
        st.error("⚠️High Risk of Heart Disease")
    else:
        st.success("✅Low Risk of Heart Disease")
          

