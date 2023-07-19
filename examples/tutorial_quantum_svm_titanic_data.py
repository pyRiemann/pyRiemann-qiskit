"""
====================================================================
Use QuanticSVM with real data from Kaggle's Titanic Dataset
====================================================================

Practical application of quantum-enhanced SVM to the titanic dataset:
https://www.kaggle.com/c/titanic/data

"""
# Author: Adrien Veres
# License: BSD (3-clause)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from pyriemann_qiskit.classification import QuanticSVM

print(__doc__)


###############################################################################
# Useful functions


def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator / denominator
    return eta


def clean(data):
    # dropping the columns that are not significant for survival
    droppable_columns = ["Name", "Ticket", "Cabin"]
    data = data.drop(droppable_columns, axis=1)

    # "Age" missing values replaced by the mean age
    data["Age"].fillna(data["Age"].mean(), inplace=True)
    # data['Age'] = data[['Age', "Pclass"]].apply(age_engineering, axis = 1)

    # "Embarked" missing values replaced by U(unknown)
    data["Embarked"].fillna("U", inplace=True)

    # "SibSp", "Parch" missing values replaced by 0
    data["SibSp"].fillna(0, inplace=True)
    data["Parch"].fillna(0, inplace=True)

    # "Fare" missing values replaced by its median
    data["Fare"].fillna(data["Fare"].mean(), inplace=True)

    return data


###############################################################################
# Download data

gender_sub = pd.read_csv("gender_submission.csv")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# # *Data Dictionary*
# * Variable	Definition	Key
# * survival	Survival	0 = No, 1 = Yes
# * pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# * sex	Sex
# * Age	Age in years
# * sibsp	# of siblings / spouses aboard the Titanic
# * parch	# of parents / children aboard the Titanic
# * ticket	Ticket number
# * fare	Passenger fare
# * cabin	Cabin number
# * embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
#
# # Variable Notes
# **pclass:** A proxy for socio-economic status (SES)
# 1. * 1st = Upper
# 1. * 2nd = Middle
# 1. * 3rd = Lower
# **age:** Age is fractional if less than 1.
#      If the age is estimated, is it in the form of xx.5
#
# **sibsp:** The dataset defines family relations in this way...
# 1. * Sibling = brother, sister, stepbrother, stepsister
# 1. * Spouse = husband, wife (mistresses and fiancés were ignored)
# **parch:** The dataset defines family relations in this way...
# 1. * Parent = mother, father
# 1. * Child = daughter, son, stepdaughter, stepson
# 1. * Some children travelled only with a nanny, therefore parch=0 for them.

###############################################################################
# Exploration

gender_sub.head()
train.head()
test.head()

# Compute dataset statistics
train.describe()

# Display missing values
print(train.isna().sum())
sns.heatmap(train.isna())

# Compute fill-ness
fill_rate = train.notnull().mean()
print(fill_rate)

print(train["Pclass"].describe())
sns.displot(x="Pclass", data=train)

print(train["Age"].describe())
sns.displot(x="Age", data=train)

print(train["Sex"].describe())
sns.displot(x="Sex", data=train)

print(train["Embarked"].describe())
sns.displot(x="Embarked", data=train)

print(train["Fare"].describe())
sns.displot(x="Fare", data=train)

sns.catplot(
    data=train,
    y="Pclass",
    hue="Survived",
    kind="count",
    palette="pastel",
    edgecolor=".6",
)

sns.boxplot(data=train, x="Pclass", y="Age", hue="Survived")

sns.boxplot(data=train, x="Age", y="Sex", hue="Survived")

sns.boxplot(data=train, x="Embarked", y="Age", hue="Survived")

sns.catplot(
    data=train,
    y="SibSp",
    hue="Survived",
    kind="count",
    palette="pastel",
    edgecolor=".6",
)

sns.catplot(
    data=train,
    y="Parch",
    hue="Survived",
    kind="count",
    palette="pastel",
    edgecolor=".6",
)

sns.catplot(
    data=train,
    y="Embarked",
    hue="Survived",
    kind="count",
    palette="pastel",
    edgecolor=".6",
)

sns.catplot(
    data=train,
    y="Sex",
    hue="Survived",
    kind="count",
    palette="pastel",
    edgecolor=".6",
)

sns.catplot(
    data=train,
    y="Pclass",
    hue="Survived",
    kind="count",
    palette="pastel",
    edgecolor=".6",
)

###############################################################################
# Imputation with KNNs (Feature Engineering)

# Select appropriate features
features = train[
    ["PassengerId", "Name", "Sex", "Age", "Fare", "Embarked", "Pclass", "Survived"]
]

features.head()

# Compute the number of survivors/deceased persons
survived_counts = train["Survived"].value_counts()
print(survived_counts)

# Survival rate is about **38.38%**

sns.countplot(x="Survived", data=train)

# Histogram before imputation
train.hist(sharex=True, sharey=True)

# Select features to impute
age_cols = ["Age", "Pclass", "SibSp", "Parch", "Fare"]

# Imput
train = pd.read_csv("train.csv")
imputer = KNNImputer(n_neighbors=5)
train[age_cols] = imputer.fit_transform(train[age_cols])

print(train.head())

# Check missing values are now fulfilled
print(train.isna().sum())

# Histogram after imputation
train.hist(sharex=True, sharey=True)


# # Compute survival rate

sex_group = train.groupby(["Sex"])["Survived"].mean()

age_group = pd.cut(train["Age"], [0, 18, 25, 40, 60, 100])
age_group = train.groupby([age_group])["Survived"].mean()

fare_group = pd.cut(train["Fare"], [0, 10, 25, 50, 100, 1000])
fare_group = train.groupby([fare_group])["Survived"].mean()

embarked_group = train.groupby(["Embarked"])["Survived"].mean()

class_group = train.groupby(["Pclass"])["Survived"].mean()

print("Survival rate by sex :\n", sex_group)
print("\nSurvival rate by age :\n", age_group)
print("\nSurvival rate by fair :\n", fare_group)
print("\nSurvival rate by embarkation point :\n", embarked_group)
print("\nSurvival rate by class :\n", class_group)

# Compute correlation matrix between Embarked & Fare

# Label-encoding of "Embarked"
le = LabelEncoder()
train_embarked_encoded = le.fit_transform(train["Embarked"])

R2 = np.corrcoef(train_embarked_encoded, train["Fare"])
print(R2)

###############################################################################
# Multivariate analysis

# # ANOVA Rule of thumb
correlation_ratio(features["Sex"], features["Survived"])

# Correlation should be average
correlation_ratio(features["Age"], features["Survived"])

# Correlation should be strong
correlation_ratio(features["Fare"], features["Survived"])

# Correlation should be weak
correlation_ratio(features["Embarked"], features["Survived"])

# Correlation should be average
correlation_ratio(features["Pclass"], features["Survived"])


# Check the correlation between fair and class

correlation_ratio(features["Fare"], features["Pclass"])

# # PCA and correlation circle

data = train

# Select features
variables = ["Age", "Pclass", "SibSp", "Parch", "Fare"]

X = data[variables].values

# Standard scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimension reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
components_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])

# Add target
components_df["Survived"] = data["Survived"]

# Display correlation circle
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axhline(0, color="gray", lw=1)
ax.axvline(0, color="gray", lw=1)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
for i, variable in enumerate(variables):
    ax.annotate(variable, (pca.components_[0, i], pca.components_[1, i]))
    ax.arrow(
        0,
        0,
        pca.components_[0, i],
        pca.components_[1, i],
        color="r",
        width=0.01,
        head_width=0.05,
    )
plt.show()


###############################################################################
# Classification

# Cleaning
train = clean(train)

# One-hot encoding
train = pd.get_dummies(train, columns=["Sex", "Embarked", "Pclass"])

# train/test validation
# (the same for all classifiers - avoid biases in comparison)
X_train, X_test, y_train, y_test = train_test_split(
    train.drop("Survived", axis=1), train["Survived"], test_size=0.2, random_state=42
)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = balanced_accuracy_score(y_test, y_pred)
print("Balanced accuracy:", accuracy)

# linear SVC

# Use same train/test set as for logistic regression
model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = balanced_accuracy_score(y_test, y_pred)
print("Balanced accuracy:", accuracy)

# SVC + RBF

model = SVC(kernel="rbf")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = balanced_accuracy_score(y_test, y_pred)
print("score:", score)


# SVM + quantum kernel

# Tackles type conversion issue with QuanticSVM
X_train2 = X_train.astype("float64")

model = QuanticSVM(quantum=True, pegasos=False)
model.fit(X_train2, y_train)

y_pred = model.predict(X_test)

score = balanced_accuracy_score(y_test, y_pred)
print("score:", score)
