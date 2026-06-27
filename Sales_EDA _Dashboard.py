import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Loading data - no download

df= sns.load_dataset("titanic")
print(df.shape) # see: (891, 15) = 891 rows, 15 columns
# print(df.head())

# 2. Clean Data - 2 quick fix

df = df.dropna(subset=['age','embarked']) #drop rows with missing age
df['age'] =df['age'].astype(int)    # Make age whole numbers
print(df.isnull().sum())    #check nulls are gone

# 3. [ making the 4 EDA plots]
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Age Distribution of Passengers')
plt.show()

# Plot 1: Bar chart - Survival by Gender
# Insight : Females survived ~ 74%, Males ~ 19%

sns.barplot(x='sex', y='survived', data=df)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate %')
plt.show()

        # Plot 2: Histogram - Age Distribution
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Age Distribution of Passengers')
plt.show()


        #Plot 3: Heatmap - Correlations
    #Insight: Pclass and Fare are Strongly correlated

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True , cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
print("="*58)
print("\n 10")

        #Plot 4: Boxplot - Fare by passenger class

sns.boxplot(x='pclass', y='fare', data=df)
plt.title('Fare Distribution By Class')
plt.show()

        # Step 4 : Write 3 Insight
    #Key Insights : '''1. Women had a 3
#               2. 1st Class passengers paid way more and were younger.
#               3.Most passengers were between 20 -40 years old'''

#Save all 4 plots to an images folder

os.makedirs('images', exist_ok=True)

#Plot 1
sns.barplot(x='pclass', y='fare', data=df)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate %')
plt.savefig('images/survival_by_gender.png', dpi=150, bbox_inches='tight')
plt.close()

#Plot 2:
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Age Distribution of Passengers')
plt.savefig('images/age_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

#Plot 3:
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True , cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('images/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

#Plot 4:
plt.figure(figsize=(6,4))
sns.boxplot(x='pclass', y='fare', data=df)
plt.title('Fare Distribution By Class')
plt.savefig('images/fare_by_class.png', dpi=150, bbox_inches='tight')
plt.close()

print("All $ images saved in /images folder")


