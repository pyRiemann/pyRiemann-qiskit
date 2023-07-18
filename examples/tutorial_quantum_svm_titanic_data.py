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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from pyriemann_qiskit.classification import (
    QuantumClassifierWithDefaultRiemannianPipeline,
)
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from pyriemann_qiskit.classification import QuanticSVM

print(__doc__)

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
# **age:** Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
#
# **sibsp:** The dataset defines family relations in this way...
# 1. * Sibling = brother, sister, stepbrother, stepsister
# 1. * Spouse = husband, wife (mistresses and fiancés were ignored)
# **parch:** The dataset defines family relations in this way...
# 1. * Parent = mother, father
# 1. * Child = daughter, son, stepdaughter, stepson
# 1. * Some children travelled only with a nanny, therefore parch=0 for them.

# Apercus des CSV
gender_sub.head()
train.head()
test.head()

# describe pour avoir un apercus
train.describe()

# voir les erreurs et valeurs manquantes
print(train.isna().sum())
sns.heatmap(train.isna())

# Calculer le taux de remplissage. Pas beaucoup d'info sur les cabines, seulement une partie sur l'age
fill_rate = train.notnull().mean()
print(fill_rate)


###############################################################################
# Imputation with KNNs (Feature Engineering)

# Sélectionner les features principales pour l'analyse
features = train[
    ["PassengerId", "Name", "Sex", "Age", "Fare", "Embarked", "Pclass", "Survived"]
]

# Imprimer un apercu tableau
features.head()

# Connaitre le nombres de survivants et de décés
survived_counts = train["Survived"].value_counts()
print(survived_counts)

# Il y a donc **549 décés** et **342 survivants**
# Le taux de décés est donc de **61.62%** , le taux de survie de **38.38%**

# **Countplot de décés vs survies**
sns.countplot(x="Survived", data=train)

# Histogramme avant nettoyage
train.hist(sharex=True, sharey=True)

# Charger les données d'entraînement
train = pd.read_csv("train.csv")

# Sélectionner les colonnes pertinentes pour l'imputation des âges manquants
age_cols = ["Age", "Pclass", "SibSp", "Parch", "Fare"]

# Créer un imputeur K-NN avec 5 voisins
imputer = KNNImputer(n_neighbors=5)

# Imputer les âges manquants
train[age_cols] = imputer.fit_transform(train[age_cols])

# Afficher les premières lignes des données d'entraînement imputées
print(train.head())

# voir si le KNN a remplacer les ages manquants
print(train.isna().sum())

# Histogramme aprés nettoyage
train.hist(sharex=True, sharey=True)


# # Calcul du taux de survie par **Sex, Age, Tarif, Port d'embarquement et par classe**

# regrouper les données par sexe et calculer les taux de survie
sex_group = train.groupby(["Sex"])["Survived"].mean()

# regrouper les données par âge et calculer les taux de survie
age_group = pd.cut(train["Age"], [0, 18, 25, 40, 60, 100])
age_group = train.groupby([age_group])["Survived"].mean()

# regrouper les données par tarif et calculer les taux de survie
fare_group = pd.cut(train["Fare"], [0, 10, 25, 50, 100, 1000])
fare_group = train.groupby([fare_group])["Survived"].mean()

# regrouper les données par port d'embarquement et calculer les taux de survie
embarked_group = train.groupby(["Embarked"])["Survived"].mean()

# regrouper les données par classe et calculer les taux de survie
class_group = train.groupby(["Pclass"])["Survived"].mean()

print("Taux de survie par sexe :\n", sex_group)
print("\nTaux de survie par âge :\n", age_group)
print("\nTaux de survie par tarif :\n", fare_group)
print("\nTaux de survie par port d'embarquement :\n", embarked_group)
print("\nTaux de survie par classe :\n", class_group)

# # Faire histogramme avant et aprés knn

# # Détails des taux de survies :
# **Taux de survie par sex :**
# 1. Femmes : 74.2%
# 1. Hommes: 18.8%
#
# **Taux de survie par âge :**
# * (0, 18]      49.31%
# * (18, 25]     33.14%
# * (25, 40]     37.16%
# * (40, 60]     39.28%
# * (60, 100]    22.72%
#
#
# **Taux de survie par tarif :**
# * (0, 10]        20.56%
# * (10, 25]       42.08%
# * (25, 50]       41.95%
# * (50, 100]      65.42%
# * (100, 1000]    73.58%
#
# **Taux de survie par port d'embarquement :**
# 1. Cherbourg     55.35%
# 1. Queenstown    38.96%
# 1. Southampton   33.69%
#
# **Taux de survie par classe :**
# 1. 1    62.96%
# 1. 2    47.28%
# 1. 3    24.23%
#

# ** Calculer la matrice de corrélation entre deux variables ( Embarked & Fare) dans un ensemble de données. **

# créer un encodeur de label
le = LabelEncoder()

# encoder la colonne "Embarked"
train_embarked_encoded = le.fit_transform(train["Embarked"])

# utiliser np.corrcoef() avec les données encodées :calculer la matrice de corrélation entre les différentes variables d'un ensemble de données
R2 = np.corrcoef(train_embarked_encoded, train["Fare"])
print(R2)

###############################################################################
# Multivariate analysis


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


def display_circles(
    pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None
):
    for (
        d1,
        d2,
    ) in (
        axis_ranks
    ):  # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:
            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7, 6))

            # détermination des limites du graphique
            if lims is not None:
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30:
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else:
                xmin, xmax, ymin, ymax = (
                    min(pcs[d1, :]),
                    max(pcs[d1, :]),
                    min(pcs[d2, :]),
                    max(pcs[d2, :]),
                )

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30:
                plt.quiver(
                    np.zeros(pcs.shape[1]),
                    np.zeros(pcs.shape[1]),
                    pcs[d1, :],
                    pcs[d2, :],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color="grey",
                )
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
                ax.add_collection(
                    LineCollection(lines, axes=ax, alpha=0.1, color="black")
                )

            # affichage des noms des variables
            if labels is not None:
                for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                        plt.text(
                            x,
                            y,
                            labels[i],
                            fontsize="14",
                            ha="center",
                            va="center",
                            rotation=label_rotation,
                            color="blue",
                            alpha=0.5,
                        )

            # affichage du cercle
            circle = plt.Circle((0, 0), 1, facecolor="none", edgecolor="b")
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color="grey", ls="--")
            plt.plot([0, 0], [-1, 1], color="grey", ls="--")

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel(
                "F{} ({}%)".format(
                    d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)
                )
            )
            plt.ylabel(
                "F{} ({}%)".format(
                    d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)
                )
            )

            plt.title("Cercle des corrélations (F{} et F{})".format(d1 + 1, d2 + 1))
            plt.show(block=False)


def display_factorial_planes(
    X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None
):
    for d1, d2 in axis_ranks:
        if d2 < n_comp:
            # initialisation de la figure
            fig = plt.figure(figsize=(7, 6))

            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(
                        X_projected[selected, d1],
                        X_projected[selected, d2],
                        alpha=alpha,
                        label=value,
                    )
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                    plt.text(x, y, labels[i], fontsize="14", ha="center", va="center")

            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
            plt.xlim([-boundary, boundary])
            plt.ylim([-boundary, boundary])

            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color="grey", ls="--")
            plt.plot([0, 0], [-100, 100], color="grey", ls="--")

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel(
                "F{} ({}%)".format(
                    d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)
                )
            )
            plt.ylabel(
                "F{} ({}%)".format(
                    d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)
                )
            )

            plt.title(
                "Projection des individus (sur F{} et F{})".format(d1 + 1, d2 + 1)
            )
            plt.show(block=False)


def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_ * 100
    plt.bar(np.arange(len(scree)) + 1, scree)
    plt.plot(np.arange(len(scree)) + 1, scree.cumsum(), c="red", marker="o")
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)


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


# # ANOVA Rule of thumb , Niveau de corrélation entre Survived et Sex ( Fort)
correlation_ratio(features["Sex"], features["Survived"])

# Corrélation Moyenne
correlation_ratio(features["Age"], features["Survived"])

# Corrélation forte
correlation_ratio(features["Fare"], features["Survived"])

# Corrélation faible
correlation_ratio(features["Embarked"], features["Survived"])

# Corrélation Moyenne
correlation_ratio(features["Pclass"], features["Survived"])


# On vérifie l'hypothese qu'il existe une trés forte corrélation entre le tarif et la classe d'un passager

# Corrélation entre le tarif et la classe d'un passager
correlation_ratio(features["Fare"], features["Pclass"])


# # PCA et Cercle de corrélation

data = train

# Sélectionner les variables à inclure dans l'analyse
variables = ["Age", "Pclass", "SibSp", "Parch", "Fare"]

# Extraire les données pour les variables sélectionnées
X = data[variables].values

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Créer un objet PCA avec 2 composantes principales
pca = PCA(n_components=2)

# Appliquer l'analyse en composantes principales aux données standardisées
principal_components = pca.fit_transform(X_scaled)

# Créer un DataFrame pour les composantes principales
components_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])

# Ajouter les données de la variable cible (Survived) au DataFrame
components_df["Survived"] = data["Survived"]

# Tracer le cercle de corrélation
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


# # Cercle de corrélation
# 1. La composante PC1 est CO linéaire et expliquée en grande partie par Parch, SibSp puis Fare. La corrélation est forte. Donc les features Parch, SibSp peuvent être utilisées pour prédire PC1.
# 1. La composante PC2 est Co linéaire et expliquée en grande partie par Pclass puis Age. La corrélation est forte. Donc les features Pclass et Age peuvent être utilisées pour prédire PC2.
# 1. PC1 et PC2 expliquent une grande partie de la variance du modele.

train.head()

# # Nettoyage


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


# voir les erreurs et valeurs manquantes : Tout semble OK
print(train.isna().sum())

train_data = train
test_data = test

print(train_data["Pclass"].describe())
sns.displot(x="Pclass", data=train_data)

print(train_data["Age"].describe())
sns.displot(x="Age", data=train_data)

print(train_data["Sex"].describe())
sns.displot(x="Sex", data=train_data)

print(train_data["Embarked"].describe())
sns.displot(x="Embarked", data=train_data)

print(train_data["Fare"].describe())
sns.displot(x="Fare", data=train_data)

sns.catplot(
    data=train_data,
    y="Pclass",
    hue="Survived",
    kind="count",
    palette="pastel",
    edgecolor=".6",
)

sns.boxplot(data=train_data, x="Pclass", y="Age", hue="Survived")

sns.boxplot(data=train_data, x="Age", y="Sex", hue="Survived")

sns.boxplot(data=train_data, x="Embarked", y="Age", hue="Survived")

sns.catplot(
    data=train_data,
    y="SibSp",
    hue="Survived",
    kind="count",
    palette="pastel",
    edgecolor=".6",
)

sns.catplot(
    data=train_data,
    y="Parch",
    hue="Survived",
    kind="count",
    palette="pastel",
    edgecolor=".6",
)

sns.catplot(
    data=train_data,
    y="Embarked",
    hue="Survived",
    kind="count",
    palette="pastel",
    edgecolor=".6",
)

sns.catplot(
    data=train_data,
    y="Sex",
    hue="Survived",
    kind="count",
    palette="pastel",
    edgecolor=".6",
)

sns.catplot(
    data=train_data,
    y="Pclass",
    hue="Survived",
    kind="count",
    palette="pastel",
    edgecolor=".6",
)


###############################################################################
# Classification

# # Logistic Regression

# Nettoyage
train = clean(train)

# Logistic Regression

# Convertir les variables catégorielles en données numériques en utilisant l'encodage one-hot
train = pd.get_dummies(train, columns=["Sex", "Embarked", "Pclass"])


# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    train.drop("Survived", axis=1), train["Survived"], test_size=0.2, random_state=42
)

# Construire un modèle de régression logistique
model = LogisticRegression()
model.fit(X_train, y_train)

# Prédire les résultats sur les données de test
y_pred = model.predict(X_test)

# Calculer la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# SVC lineaire

model = SVC()
model.fit(X_train, y_train)

# Prédire les résultats sur les données de test
y_pred = model.predict(X_test)

# Calculer la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# SVC + RBF

# Construire un modèle SVM avec noyau RBF
model = SVC(kernel="rbf")
model.fit(X_train, y_train)

# Prédire les résultats sur les données de test
y_pred = model.predict(X_test)

# Calculer la précision du modèle
score = balanced_accuracy_score(y_test, y_pred)
print("score:", score)

X_train.dtypes


# **On Créer X_train2 pour mettre les données de X_train dans le même type "float64"**

X_train2 = X_train.astype("float64")

X_train2.dtypes

y_train.dtypes


# Construire un modèle SVM avec noyau quantique

model = QuanticSVM(quantum=True, pegasos=False)
model.fit(X_train2, y_train)

# Prédire les résultats sur les données de test
y_pred = model.predict(X_test)

# Calculer la précision du modèle
score = balanced_accuracy_score(y_test, y_pred)
print("score:", score)
