import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data/employee_data.csv')

print("Dataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# -------------------------------
# 1. Distribution of Performance
# -------------------------------
sns.countplot(x='Performance', data=df)
plt.title("Employee Performance Distribution")
plt.savefig("outputs/performance_distribution.png")
plt.show()

# -------------------------------
# 2. Salary vs Performance
# -------------------------------
sns.boxplot(x='Performance', y='Salary', data=df)
plt.title("Salary vs Performance")
plt.savefig("outputs/salary_vs_performance.png")
plt.show()

# -------------------------------
# 3. Experience vs Performance
# -------------------------------
sns.boxplot(x='Performance', y='Experience', data=df)
plt.title("Experience vs Performance")
plt.savefig("outputs/experience_vs_performance.png")
plt.show()

# -------------------------------
# 4. Correlation Heatmap
# -------------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

# -------------------------------
# 5. Training Hours vs Performance
# -------------------------------
sns.boxplot(x='Performance', y='Training_Hours', data=df)
plt.title("Training Hours vs Performance")
plt.savefig("outputs/training_vs_performance.png")
plt.show()

print("EDA Completed!")