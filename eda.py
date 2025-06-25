import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


df = pd.read_csv("Titanic-Dataset.csv") 
print("Dataset loaded successfully.\n")


#Creating output folder for plots
if not os.path.exists("plots"):
    os.makedirs("plots")

#Summary Statistics
print("Summary Statistics:")
print(df.describe(include='all'))

print("Missing Values:")
print(df.isnull().sum())

#Histograms for numeric columns
print("\nGenerating histograms...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Histogram of {col}")
    plt.tight_layout()
    plt.savefig(f"plots/hist_{col}.png")
    plt.close()

#Boxplots for numeric columns
print("Generating boxplots...")
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(f"plots/box_{col}.png")
    plt.close()

#Pairplot for selected features
pair_cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'Survived']
pair_cols = [col for col in pair_cols if col in df.columns]

if len(pair_cols) >= 2:
    print("🔗 Generating pairplot...")
    sns.pairplot(df[pair_cols].dropna())
    plt.savefig("plots/pairplot.png")
    plt.close()

#Correlation Matrix
print("Generating correlation heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

print("\nAll EDA visualizations saved in the 'plots/' folder.")
