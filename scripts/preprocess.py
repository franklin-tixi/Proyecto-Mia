# =========================
# 1. LIBRERÍAS
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================================
# 2. CARGA DE DATOS
# ==========================================================================

df = pd.read_excel('/content/DataSetMIA.xlsx')
print("Archivo cargado exitosamente...!")

# ==========================================================================
# CONVERTIR GL DATE
# ==========================================================================

df['GL Date'] = pd.to_numeric(df['GL Date'], errors='coerce')
df['fecha'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df['GL Date'], unit='D')

# ==========================================================================
# 3. LIMPIEZA
# ==========================================================================

columnas = df.columns.tolist()
print(columnas)

df.columns = df.columns.str.strip()
df = df.rename(columns={'JDEAccounts.BUDGET LOCATION': 'Location'})
df = df.rename(columns={'Saldo': 'Costo'})


# ==========================================================================
# 4. FEATURE ENGINEERING
# ==========================================================================

df = df.sort_values(['Account', 'Year', 'Period'])
df['Lag_1'] = df.groupby('Account')['Costo'].shift(1)
df['Lag_2'] = df.groupby('Account')['Costo'].shift(2)
df['Rolling_mean_3'] = df.groupby('Account')['Costo'].transform(
    lambda x: x.rolling(3).mean()
)
df['Year_Period'] = df['Year'] * 100 + df['Period']
