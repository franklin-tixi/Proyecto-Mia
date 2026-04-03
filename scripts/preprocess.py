import pandas as pd

df = pd.read_excel("data/raw/Data Set MIA.xlsx")

df = df[['Year', 'Period', 'Saldo']].copy()
df = df.dropna()

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Period'] = pd.to_numeric(df['Period'], errors='coerce')
df['Saldo'] = pd.to_numeric(df['Saldo'], errors='coerce')

df = df.dropna()

df['Year'] = df['Year'].astype(int)
df['Period'] = df['Period'].astype(int)

df_grouped = df.groupby(['Year', 'Period'], as_index=False)['Saldo'].sum()

df_grouped['time_real'] = df_grouped['Year'] + (df_grouped['Period'] - 1) / 12
base_time = df_grouped['time_real'].min()
df_grouped['time'] = df_grouped['time_real'] - base_time

df_grouped.to_csv("data/processed/saldo_mensual.csv", index=False)

print(df_grouped.head())
