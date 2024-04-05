#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

def plot_data_fit_and_correlation(df, brand, X, y, modelo, y_pred, corr_coef, r_squared):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Datos reales')
    plt.plot(X, y_pred, color='red', label='Ajuste lineal')
    plt.title(f'Marca: {brand} - Correlación: {corr_coef:.2f}, $R^2$: {r_squared:.2f}')
    plt.xlabel('TDPs')
    plt.ylabel('Value Share')
    plt.legend()
    st.pyplot(plt)

def main():
    st.title("Análisis de Share de Valor y TDPs")

    uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)

        marcas = df['Marca'].unique()
        marca_seleccionada = st.selectbox("Selecciona una marca", marcas)

        if marca_seleccionada:
            data_marca = df[df['Marca'] == marca_seleccionada]
            X = data_marca[['TDPs']].values.reshape(-1, 1)
            y = data_marca['Value share'].values

            modelo = LinearRegression().fit(X, y)
            y_pred = modelo.predict(X)

            corr_coef, _ = pearsonr(data_marca['TDPs'], data_marca['Value share'])
            r_squared = modelo.score(X, y)

            plot_data_fit_and_correlation(df, marca_seleccionada, X, y, modelo, y_pred, corr_coef, r_squared)

if __name__ == "__main__":
    main()

