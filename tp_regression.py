# Importation des bibliothèques nécessaires
import pandas as pd              # manipulation des données
import numpy as np               # calcul numérique
import matplotlib.pyplot as plt  # visualisation
import seaborn as sns            # visualisation avancée

# Affichage des graphiques directement dans Jupyter
%matplotlib inline
# Charger le dataset (consommation des voitures)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"

dataSet = pd.read_csv(url)

# Renommer les colonnes pour correspondre au TP
dataSet = dataSet.rename(columns={
    "mpg": "MPG",
    "cylinders": "Cylinders",
    "displacement": "Displacement",
    "horsepower": "Horsepower",
    "weight": "Weight",
    "acceleration": "Acceleration",
    "model_year": "Model Year",
    "origin": "Origin"
})

# Supprimer les valeurs manquantes
dataSet = dataSet.dropna()

# Afficher les 5 premières lignes
dataSet.head()
# Afficher le nombre de lignes et de colonnes
dataSet.shape
# Vérifier les valeurs nulles
dataSet.isnull().sum()
# Visualiser les relations entre toutes les variables
sns.pairplot(dataSet)

plt.show()
# Histogramme de la variable cible MPG
sns.histplot(dataSet["MPG"], kde=True)

plt.xlabel("MPG")
plt.ylabel("Nombre")# Visualiser les relations entre toutes les variables

plt.show()
# Calcul de la matrice de corrélation
cor = dataSet.corr(numeric_only=True)

# Afficher la matrice
cor
# Visualisation de la corrélation avec heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(cor, annot=True, cmap="coolwarm")

plt.show()
from sklearn.model_selection import train_test_split

# Variables explicatives (X)
X = dataSet.drop("MPG", axis=1)

# Transformer la variable "Origin" en variables numériques
X = pd.get_dummies(X, drop_first=True)

# Variable cible (Y)
Y = dataSet["MPG"]

# Division en données d'entraînement et de test (70% / 30%)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=20
)
X_train.head()
X_train.dtypes
from sklearn.linear_model import LinearRegression
# Création du modèle de régression linéaire
lin_model = LinearRegression()

# Entraînement
lin_model.fit(X_train, Y_train)

# Prédictions
predictions = lin_model.predict(X_test)
# Comparaison valeurs réelles vs prédictions
plt.scatter(Y_test, predictions)

plt.xlabel("Valeurs réelles MPG")
plt.ylabel("Prédictions MPG")

plt.show()
# Charger le dataset (admission des étudiants)
url2 = "https://stats.idre.ucla.edu/stat/data/binary.csv"

dataset = pd.read_csv(url2)

# Renommer les colonnes
dataset.columns = ["admit", "gre", "gpa", "prestige"]

dataset.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dummy_ranks = pd.get_dummies(dataset["prestige"], prefix="prestige")

data = dataset[["admit", "gre", "gpa"]].join(dummy_ranks)

x = data.drop("admit", axis=1)
y = data["admit"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1
)

logmodel = LogisticRegression(C=1e20, max_iter=1000)
logmodel.fit(x_train, y_train)

y_pred = logmodel.predict(x_test)

count_misclassified = (y_test != y_pred).sum()
accuracy = accuracy_score(y_test, y_pred)

print("Échantillons mal classés :", count_misclassified)
print("Précision :", round(accuracy, 2))
dataset.shape
dataset.isnull().sum()
# Visualisation des relations
sns.pairplot(dataset)

plt.show()
# Transformer "prestige" en variables numériques
dummy_ranks = pd.get_dummies(dataset["prestige"], prefix="prestige")

# Créer un nouveau dataset
data = dataset[["admit", "gre", "gpa"]].join(dummy_ranks)

data.head()
# Nombre d'étudiants admis / non admis
data["admit"].value_counts()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Variables
x = data.drop("admit", axis=1)
y = data["admit"]

# Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1
)

# Création du modèle
logmodel = LogisticRegression(C=1e20, max_iter=1000)

# Entraînement
logmodel.fit(x_train, y_train)

# Prédiction
y_pred = logmodel.predict(x_test)
# Nombre d'erreurs
count_misclassified = (y_test != y_pred).sum()

# Précision
accuracy = accuracy_score(y_test, y_pred)

print("Échantillons mal classés :", count_misclassified)
print("Précision :", round(accuracy, 2))