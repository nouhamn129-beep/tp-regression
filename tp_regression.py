import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# =========================
# PARTIE 1 : REGRESSION LINEAIRE (Auto MPG depuis UCI)
# =========================

from ucimlrepo import fetch_ucirepo

auto_mpg = fetch_ucirepo(id=9)

X_auto = auto_mpg.data.features.copy()
y_auto = auto_mpg.data.targets.copy()

data_auto = pd.concat([y_auto, X_auto], axis=1)

# Remove non-numeric column if exists
if "car_name" in data_auto.columns:
    data_auto = data_auto.drop(columns=["car_name"])

print("===== PARTIE 1 : REGRESSION LINEAIRE =====")
print(data_auto.head())
print("\nShape:", data_auto.shape)

# Remove missing values
data_auto = data_auto.dropna()

X = data_auto.drop(columns=["mpg"])
y = data_auto["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=20
)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

print("===== PARTIE 1 : REGRESSION LINEAIRE =====")

# Chargement du dataset auto-mpg
data_auto = pd.read_csv("datasets/auto-mpg.csv")

print("\nAperçu du dataset auto-mpg :")
print(data_auto.head())

print("\nShape:", data_auto.shape)

# Variables explicatives
X = data_auto[[
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin"
]]

# Variable cible
y = data_auto["mpg"]

# Séparation train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=20
)

# Création et entraînement du modèle
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Prédiction sur le test set
pred_lin = lin_model.predict(X_test)

print("\nR2 score:", round(r2_score(y_test, pred_lin), 2))

# Exemple de prédiction
sample = [[8, 307, 130, 3000, 12.0, 70, 1]]
print("Prediction example MPG:", lin_model.predict(sample))

print("\n===== PARTIE 2 : REGRESSION LOGISTIQUE =====")

# Chargement du dataset binary
data_bin = pd.read_csv("datasets/binary.csv")

print("\nAperçu du dataset binary :")
print(data_bin.head())

print("\nShape:", data_bin.shape)

# Variables explicatives
X2 = data_bin[["feature1", "feature2"]]

# Variable cible
y2 = data_bin["target"]

# Séparation train / test
x_train, x_test, y_train, y_test = train_test_split(
    X2, y2, test_size=0.3, random_state=1
)

# Création et entraînement du modèle logistique
log_model = LogisticRegression()
log_model.fit(x_train, y_train)

# Prédiction sur le test set
pred_log = log_model.predict(x_test)

print("\nAccuracy:", round(accuracy_score(y_test, pred_log), 2))

# Exemple de prédiction
print("Prediction pour [3,3] :", log_model.predict([[3, 3]]))