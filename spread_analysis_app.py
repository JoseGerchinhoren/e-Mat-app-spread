import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Título de la aplicación
st.title('Análisis de Spreads de Instrumentos Agrícolas')

# Cargar datos
@st.cache_data
def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    return df

df = cargar_datos('closing_prices_limpio.csv')

# Mostrar las primeras filas del DataFrame
st.write('Datos cargados:')
st.write(df)

# Selección de productos
productos = df['PRODUCTO'].unique()
producto1 = st.selectbox('Selecciona el primer producto', productos)

# Filtrar posiciones para el primer producto
posiciones_producto1 = df[df['PRODUCTO'] == producto1]['TIPO CONTRATO'].unique()
posicion1 = st.selectbox('Selecciona la primera posición', posiciones_producto1)

# Selección del segundo producto
producto2 = st.selectbox('Selecciona el segundo producto', productos)

# Filtrar posiciones para el segundo producto
posiciones_producto2 = df[df['PRODUCTO'] == producto2]['TIPO CONTRATO'].unique()
posicion2 = st.selectbox('Selecciona la segunda posición', posiciones_producto2)

# Filtrar datos para los productos y posiciones seleccionados
df_pos1 = df[(df['PRODUCTO'] == producto1) & (df['TIPO CONTRATO'] == posicion1)]
df_pos2 = df[(df['PRODUCTO'] == producto2) & (df['TIPO CONTRATO'] == posicion2)]

# Unir dataframes por fecha
df_merged = pd.merge(df_pos1, df_pos2, on='FECHA', suffixes=('_pos1', '_pos2'))

# Calcular spread
df_merged['SPREAD'] = df_merged['AJUSTE / PRIMA REF._pos1'] - df_merged['AJUSTE / PRIMA REF._pos2']

# Mostrar los datos filtrados
st.write('Datos del primer producto y posición seleccionados:')
st.write(df_pos1.head())

st.write('Datos del segundo producto y posición seleccionados:')
st.write(df_pos2.head())

# Mostrar el DataFrame combinado y el spread
st.write('Datos combinados:')
st.write(df_merged.head())

# Crear gráfico de líneas
fig, ax = plt.subplots()
ax.plot(df_merged['FECHA'], df_merged['AJUSTE / PRIMA REF._pos1'], label=f'{producto1} {posicion1}')
ax.plot(df_merged['FECHA'], df_merged['AJUSTE / PRIMA REF._pos2'], label=f'{producto2} {posicion2}')
# ax.plot(df_merged['FECHA'], df_merged['SPREAD'], label='Spread', linestyle='--')

# Formatear el gráfico
ax.set_xlabel('Fecha')
ax.set_ylabel('Precio de ajuste / prima ref.')
ax.legend()
plt.xticks(rotation=45)

# Mostrar gráfico en Streamlit
st.pyplot(fig)

# Mostrar tabla de spreads
st.write('Tabla de spreads:')
st.write(df_merged[['FECHA', 'AJUSTE / PRIMA REF._pos1', 'AJUSTE / PRIMA REF._pos2', 'SPREAD']])
