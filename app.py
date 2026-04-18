import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv('data/employee_data.csv')

# Encode data
le = LabelEncoder()
df['Department'] = le.fit_transform(df['Department'])
df['Performance'] = le.fit_transform(df['Performance'])

# Train model
X = df.drop('Performance', axis=1)
y = df['Performance']

model = RandomForestClassifier()
model.fit(X, y)

# -------------------------------
# UI Design
# -------------------------------

st.title("💼 Employee Performance Predictor")

st.write("Enter employee details below:")

# Input fields
age = st.slider("Age", 20, 60, 30)
experience = st.slider("Experience (Years)", 0, 20, 5)
salary = st.number_input("Salary", 20000, 150000, 50000)
training = st.slider("Training Hours", 0, 100, 40)
projects = st.slider("Projects Completed", 1, 10, 4)
attendance = st.slider("Attendance (%)", 50, 100, 85)

department = st.selectbox("Department", ["HR", "IT", "Sales"])

# Convert department
dept_map = {"HR": 0, "IT": 1, "Sales": 2}
department = dept_map[department]

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Performance"):

    input_data = pd.DataFrame({
        'Age': [age],
        'Experience': [experience],
        'Salary': [salary],
        'Training_Hours': [training],
        'Projects': [projects],
        'Attendance': [attendance],
        'Department': [department]
    })

    prediction = model.predict(input_data)

    labels = ['Low', 'Medium', 'High']
    result = labels[prediction[0]]

    st.success(f"Predicted Performance: {result}")