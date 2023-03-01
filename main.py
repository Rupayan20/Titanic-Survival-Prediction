# Required Libraries for the Analysis of the Titanic Disaster
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# Data Preprocessing and Exploratory Data Analysis
train_df = pd.read_csv("https://drive.google.com/file/d/1_HsxBaBIbWIwEnGRkHhcS0MeJH355xun/view?usp=sharing")
train_df.head(10)
targets = train_df.iloc[:, 1]
print(targets)

corr_matrix = train_df.corr()
corr_matrix

plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, xticklabels=True, yticklabels=True, cmap='inferno')
corr_matrix["Survived"].sort_values(ascending=False)
train_df.info()

train_df["Embarked"] = train_df["Embarked"].fillna("S")
train_df.info()


# Barplot for identifying the probabilty of the survival of an individual depending upon the Sex, Male of Female.
plt.figure(figsize=(5, 5))
sns.barplot(
    data=train_df, 
    x = "Sex", 
    y = "Survived"
)
plt.show()


# Countplot for visualization of the total number of individual that had embarked, depending upon the dataset.
# Embarked ID are S, C and Q (Southampton, Cherbourg, and Queenstown).
plt.figure(figsize=(5, 5))
sns.countplot(x = train_df["Embarked"])

plt.figure(figsize=(5, 5))
sns.barplot(
    data = train_df, 
    x = "Embarked",
    y = "Survived",
)
plt.show()
