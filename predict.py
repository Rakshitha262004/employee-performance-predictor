import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('data/employee_data.csv')

# Encode
le = LabelEncoder()
df['Department'] = le.fit_transform(df['Department'])
df['Performance'] = le.fit_transform(df['Performance'])

# Train model again (simple way)
X = df.drop('Performance', axis=1)
y = df['Performance']

model = RandomForestClassifier()
model.fit(X, y)

# -------------------------------
# NEW EMPLOYEE INPUT
# -------------------------------
new_employee = pd.DataFrame({
    'Age': [30],
    'Experience': [5],
    'Salary': [50000],
    'Training_Hours': [40],
    'Projects': [4],
    'Attendance': [85],
    'Department': [1]  # IT
})

# Prediction
prediction = model.predict(new_employee)

# Decode result
labels = ['Low', 'Medium', 'High']
print("Predicted Performance:", labels[prediction[0]])