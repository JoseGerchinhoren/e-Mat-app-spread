import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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

# Crear dos columnas
col1, col2 = st.columns(2)

# Selección de productos en la primera columna
with col1:
    producto1 = st.selectbox('Selecciona el primer producto', df['PRODUCTO'].unique())
    posiciones_producto1 = df[df['PRODUCTO'] == producto1]['TIPO CONTRATO'].unique()
    posicion1 = st.selectbox('Selecciona la primera posición', posiciones_producto1)

# Selección del segundo producto en la segunda columna
with col2:
    producto2 = st.selectbox('Selecciona el segundo producto', df['PRODUCTO'].unique())
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

# Crear gráfico interactivo con Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_merged['FECHA'], y=df_merged['AJUSTE / PRIMA REF._pos1'], 
                         mode='lines+markers', name=f'{producto1} {posicion1}'))

fig.add_trace(go.Scatter(x=df_merged['FECHA'], y=df_merged['AJUSTE / PRIMA REF._pos2'], 
                         mode='lines+markers', name=f'{producto2} {posicion2}'))

fig.update_layout(
    xaxis_title='Fecha',
    yaxis_title='Precio de ajuste / prima ref.',
    xaxis=dict(
        tickformat='%m/%Y',
        tickmode='auto'
    ),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# Asegurarse de que la fecha completa se muestre en los puntos
fig.update_traces(
    hovertemplate='Fecha: %{x|%d/%m/%Y}<br>Precio: %{y}'
)

# Mostrar gráfico en Streamlit
st.plotly_chart(fig)

# Mostrar tabla de spreads
st.write('Tabla de spreads:')
st.write(df_merged[['FECHA', 'AJUSTE / PRIMA REF._pos1', 'AJUSTE / PRIMA REF._pos2', 'SPREAD']])
