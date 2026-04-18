import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

data = pd.DataFrame({
    'Age': np.random.randint(22, 60, n),
    'Experience': np.random.randint(1, 20, n),
    'Salary': np.random.randint(20000, 150000, n),
    'Training_Hours': np.random.randint(10, 100, n),
    'Projects': np.random.randint(1, 10, n),
    'Attendance': np.random.randint(60, 100, n),
    'Department': np.random.choice(['HR', 'IT', 'Sales'], n)
})

# Create performance score
score = (
    data['Experience'] * 2 +
    data['Training_Hours'] * 0.5 +
    data['Projects'] * 3 +
    data['Attendance'] * 0.3
)

data['Performance'] = pd.cut(score,
                             bins=[0, 100, 200, 400],
                             labels=['Low', 'Medium', 'High'])

data.to_csv('data/employee_data.csv', index=False)

print("Dataset created!")