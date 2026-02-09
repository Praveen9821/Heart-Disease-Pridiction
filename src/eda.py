import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../data/heart.csv")

print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns)

print("\nTarget Value Counts:")
print(df['target'].value_counts())

# ----------- Visualizations -----------

# 1. Target Distribution
plt.figure()
sns.countplot(x='target', data=df)
plt.title("Target Distribution (Heart Disease)")
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Use source venv/Scripts/activate to activate venv