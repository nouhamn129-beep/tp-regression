import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

print("===== PARTIE 1 : REGRESSION LINEAIRE =====")

# =========================
# 1) REGRESSION LINEAIRE
# =========================
df = pd.read_csv("datasets/auto-mpg.csv")

print("\nAperçu du dataset auto-mpg :")
print(df.head())

# X = variable explicative
X = df[["weight"]]

# y = variable cible
y = df["mpg"]

# modèle linéaire
model_lin = LinearRegression()
model_lin.fit(X, y)

# prédiction
prediction_lin = model_lin.predict(pd.DataFrame([[3000]], columns=["weight"]))

print("\nPrediction pour weight = 3000 :")
print(prediction_lin)

print("\n========================================")
print("===== PARTIE 2 : REGRESSION LOGISTIQUE =====")

# =========================
# 2) REGRESSION LOGISTIQUE
# =========================
df2 = pd.read_csv("datasets/binary.csv")

print("\nAperçu du dataset binary :")
print(df2.head())

# X = variables explicatives
X2 = df2[["feature1", "feature2"]]

# y = variable cible
y2 = df2["target"]

# modèle logistique
model_log = LogisticRegression()
model_log.fit(X2, y2)

# prédiction
prediction_log = model_log.predict(pd.DataFrame([[3, 3]], columns=["feature1", "feature2"]))

print("\nPrediction pour [3,3] :")
print(prediction_log)