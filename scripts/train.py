import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/saldo_mensual.csv")

X = df[['time']]
y = df['Saldo']

# REGRESION LINEAL
lr = LinearRegression()
lr.fit(X, y)

# RANDOM FOREST
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y)

# RED NEURONAL
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
nn.fit(X_scaled, y_scaled)

print("Modelos entrenados")
