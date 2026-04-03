import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/saldo_mensual.csv")

X = df[['time']]
y = df['Saldo']

# ENTRENAR OTRA VEZ (igual que tu notebook)
lr = LinearRegression()
lr.fit(X, y)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
nn.fit(X_scaled, y_scaled)

# PROYECCION (TU MISMO CODIGO)
base_time = df['time_real'].min()

years = np.arange(df['Year'].min(), 2036)
periods = np.arange(1, 13)

future_rows = []
for year in years:
    for period in periods:
        future_rows.append([year, period])

future = pd.DataFrame(future_rows, columns=['Year', 'Period'])

future['time_real'] = future['Year'] + (future['Period'] - 1) / 12
future['time'] = future['time_real'] - base_time

pred_lr = lr.predict(future[['time']])
pred_rf = rf.predict(future[['time']])

future_scaled = scaler_X.transform(future[['time']])
pred_nn_scaled = nn.predict(future_scaled)
pred_nn = scaler_y.inverse_transform(pred_nn_scaled.reshape(-1, 1)).ravel()

resultado = future[['Year', 'Period', 'time_real']].copy()
resultado['Pred_LR'] = pred_lr
resultado['Pred_RF'] = pred_rf
resultado['Pred_NN'] = pred_nn

print(resultado.head())
