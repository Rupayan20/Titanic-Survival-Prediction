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

# Actual dataset prepration begins here. The Data tensors are extracted from the actual training dataset.
seed = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"]
df_train_set = train_df.drop(seed, axis=1)
df_train_set.head(10)

mean_age = df_train_set["Age"].mean()
mean_fare = df_train_set["Fare"].mean()

df_train_set["Age"], df_train_set["Fare"] = df_train_set["Age"].fillna(mean_age), df_train_set["Fare"].fillna(mean_fare)

df_train_set.info()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df_train_set.iloc[:, 2] = encoder.fit_transform(
    df_train_set.iloc[:, 2].values
)
df_train_set.iloc[:, 5] = encoder.fit_transform(
    df_train_set.iloc[:, 5].values
)

df_train_set.info()

df_train_set.head(10)

X = df_train_set.iloc[:, 1:6].values

y = df_train_set.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("The Shape of the training data tensor are : {} for the features and {} for the targets.".format(X_train.shape, y_train.shape))
print("The Shape of the validation data tensor are : {} for the features and {} for the targets.".format(X_test.shape, y_test.shape))


# Implementation of the K-Nearest Neighbors Classifier Algorithm for prediction of passenger survivality.
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from tqdm import tqdm

k_range = range(1, 500)
scores = {}
scores_list = []
for k in tqdm(k_range) :
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_prediction = knn_model.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test, y_prediction)
    scores_list.append(metrics.accuracy_score(y_test, y_prediction))
    
from matplotlib import pyplot as plt
plt.figure(figsize=(5, 5))
plt.plot(k_range, scores_list)
plt.xlabel("Value of K for the KNN Classifier")
plt.ylabel("Testing Accuracy")

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

output_prediction = knn.predict(X_test)

classes = {0 : "Will not survive", 1 : "Will survive"}

# Features for testing are :
# Pclass (1, 2, 3)	|  Sex (0, 1)  |  Age (Float Number)  |  Fare (Float Number)  |  Embarked (0, 1, 2)

X_new_samples = [
    [3, 1, 55.25, 8.25, 1], 
    [2, 1, 12.25, 6.15, 0], 
    [0, 1, 75.25, 22.25, 0],
    [1, 0, 39.25, 74.65, 0],
]

y_predict = knn.predict(X_new_samples)
print([classes[y_predict[i]] for i in range(len(y_predict))])

accuracy = metrics.accuracy_score(y_test, output_prediction)
print("The Accuracy of the Trained KNN Classifier is : {} % .".format(accuracy * 100))
