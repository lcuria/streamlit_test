#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Función para cargar los datos desde un archivo Excel
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        return None

# Función para plotear los ajustes de regresión lineal y la correlación por marca
def plot_data_fit_and_correlation(data_marca, marca):
    X = data_marca[['TDPs']].values.reshape(-1, 1)
    y = data_marca['Value share'].values

    modelo = LinearRegression().fit(X, y)
    y_pred = modelo.predict(X)

    corr_coef, _ = pearsonr(data_marca['TDPs'], data_marca['Value share'])
    r_squared = modelo.score(X, y)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Datos reales')
    plt.plot(X, y_pred, color='red', label='Ajuste lineal')
    plt.title(f'Marca: {marca} - Correlación: {corr_coef:.2f}, $R^2$: {r_squared:.2f}')
    plt.xlabel('TDPs')
    plt.ylabel('Value Share')
    plt.legend()
    st.pyplot(plt)

# Función para plotear las tendencias agregadas de Value Share y TDPs por Mes
def plot_aggregate_trends(df):
    df_grouped_by_month = df.groupby('Mes', sort=False).sum()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Mes')
    ax1.set_ylabel('Value Share', color=color)
    ax1.plot(df_grouped_by_month.index, df_grouped_by_month['Value share'], color=color, marker='o', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('TDPs', color=color)
    ax2.plot(df_grouped_by_month.index, df_grouped_by_month['TDPs'], color=color, marker='x', linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    st.pyplot(fig)

# Función para predecir la necesidad de TDPs basada en los objetivos de Value Share
def predict_tdps(df, value_share_targets):
    predictions = {}
    tdps_dic = df[df['Mes'] == 'Dic'].set_index('Marca')['TDPs']

    for brand, target in value_share_targets.items():
        brand_data = df[df['Marca'] == brand]
        X_brand = brand_data[['Value share']].values.reshape(-1, 1)
        y_brand = brand_data['TDPs'].values
        
        if len(X_brand) > 0 and len(y_brand) > 0:
            model_brand = LinearRegression().fit(X_brand, y_brand)
            predicted_tdps_for_target = model_brand.predict([[target]])[0]
            tdps_difference = predicted_tdps_for_target - tdps_dic.get(brand, 0)
            
            predictions[brand] = tdps_difference
        else:
            predictions[brand] = None
            
    return predictions

# Configuración de la página de Streamlit
st.title('Análisis y Previsión de TDPs y Value Share')

# Widget de carga de archivo
uploaded_file = st.file_uploader("Cargar archivo de datos Excel", type=["xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        marcas = df['Marca'].unique()
        marca_seleccionada = st.selectbox('Selecciona una Marca para Analizar', marcas)
        
        if marca_seleccionada:
            data_marca = df[df['Marca'] == marca_seleccionada]
            plot_data_fit_and_correlation(data_marca, marca_seleccionada)

        if st.checkbox('Mostrar Tendencias Agregadas'):
            plot_aggregate_trends(df)
        
        st.subheader("Cargar Objetivos de Share para Predicción de TDPs")
        value_share_targets_input = st.text_area("Ingresar objetivos de Share por marca en formato CSV (marca,objetivo):", 
                                                 "Dove,8.3\nSedal,14\nSuave,0.5\nTSM,5", 
                                                 height=100)
        if st.button('Calcular Necesidades de TDPs y Total Share Target'):
            value_share_targets = {line.split(',')[0]: float(line.split(',')[1]) for line in value_share_targets_input.split('\n') if line}
            predictions = predict_tdps(df, value_share_targets)
            
            st.write("Necesidades de TDPs por marca para alcanzar los objetivos de Share:")
            for brand, tdp in predictions.items():
                st.write(f"{brand}: {tdp:.2f}" if tdp is not None else f"{brand}: Predicción no disponible")
            
            # Calcular y mostrar el total de Share Target
            total_share_target = sum(value_share_targets.values())
            st.write(f"Total Share Target: {total_share_target}")



