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

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# =====================================================
# PARTIE 1 : REGRESSION LINEAIRE
# =====================================================

print("===== PARTIE 1 : REGRESSION LINEAIRE =====")

# Charger dataset auto-mpg
data_auto = pd.read_csv("datasets/auto-mpg.csv")

print("\nAperçu du dataset auto-mpg :")
print(data_auto.head())

print("\nShape:", data_auto.shape)

# X (features)
X = data_auto[[
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin"
]]

# y (target)
y = data_auto["mpg"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=20
)

# Model
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Prediction
pred_lin = lin_model.predict(X_test)

print("\nR2 score:", round(r2_score(y_test, pred_lin), 2))

# Exemple prediction
sample = [[8, 307, 130, 3000, 12.0, 70, 1]]
print("Prediction example MPG:", lin_model.predict(sample))


# =====================================================
# PARTIE 2 : REGRESSION LOGISTIQUE
# =====================================================

print("\n===== PARTIE 2 : REGRESSION LOGISTIQUE =====")

# Charger dataset binary
data_bin = pd.read_csv("datasets/binary.csv")

print("\nAperçu du dataset binary :")
print(data_bin.head())

print("\nShape:", data_bin.shape)

# X (features)
X2 = data_bin[["feature1", "feature2"]]

# y (target)
y2 = data_bin["target"]

# Split train/test
x_train, x_test, y_train, y_test = train_test_split(
    X2, y2, test_size=0.3, random_state=1
)

# Model
log_model = LogisticRegression()
log_model.fit(x_train, y_train)

# Prediction
pred_log = log_model.predict(x_test)

print("\nAccuracy:", round(accuracy_score(y_test, pred_log), 2))

# Exemple prediction
print("Prediction pour [3,3] :", log_model.predict([[3, 3]]))