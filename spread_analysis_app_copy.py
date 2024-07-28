import streamlit as st
import pandas as pd
import boto3
import io
from datetime import datetime
import plotly.graph_objects as go

# Obtener credenciales desde el archivo config
from config import cargar_configuracion

# Conectar a S3
def conectar_s3():
    aws_access_key, aws_secret_key, region_name, bucket_name = cargar_configuracion()
    return boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=region_name), bucket_name

def cargar_dataframe_desde_s3(s3, bucket_name, archivo_csv):
    try:
        response = s3.get_object(Bucket=bucket_name, Key=archivo_csv)
        return pd.read_csv(io.BytesIO(response['Body'].read()))
    except s3.exceptions.NoSuchKey:
        st.warning("No se encontró el archivo CSV en S3.")
        return pd.DataFrame(columns=['PRODUCTO', 'TIPO CONTRATO', 'AJUSTE / PRIMA REF.', 'AÑO', 'MES-DIA'])

# Título de la aplicación
st.title('Spread de Instrumentos Agrícolas')

# Conectar a S3 y cargar datos
s3, bucket_name = conectar_s3()
archivo_csv = 'historico_spread.csv'
df = cargar_dataframe_desde_s3(s3, bucket_name, archivo_csv)

# Asegurarse de que los datos se cargaron correctamente
if df.empty:
    st.error("No se pudieron cargar los datos. Verifica el archivo en S3.")
else:
    # Verificar y limpiar datos en la columna TIPO CONTRATO
    def limpiar_tipo_contrato(tipo_contrato):
        try:
            return int(tipo_contrato[-2:])
        except ValueError:
            # En caso de error, devolver un valor que no afecte el orden
            return float('inf')

    df['TIPO_CONTRATO_CLEAN'] = df['TIPO CONTRATO'].apply(limpiar_tipo_contrato)

    # Crear dos columnas
    col1, col2 = st.columns(2)

    # Selección de productos en la primera columna
    with col1:
        producto1 = st.selectbox('Selecciona el primer producto', df['PRODUCTO'].unique())
        posiciones_producto1 = sorted(df[df['PRODUCTO'] == producto1]['TIPO CONTRATO'].unique(), key=lambda x: (limpiar_tipo_contrato(x), x.split('/')[0]), reverse=True)
        posicion1 = st.selectbox('Selecciona la primera posición', posiciones_producto1)

    # Selección del segundo producto en la segunda columna
    with col2:
        producto2 = st.selectbox('Selecciona el segundo producto', df['PRODUCTO'].unique())
        posiciones_producto2 = sorted(df[df['PRODUCTO'] == producto2]['TIPO CONTRATO'].unique(), key=lambda x: (limpiar_tipo_contrato(x), x.split('/')[0]), reverse=True)
        posicion2 = st.selectbox('Selecciona la segunda posición', posiciones_producto2)

    # Filtrar datos de los productos y posiciones seleccionadas
    df_filtro1 = df[(df['PRODUCTO'] == producto1) & (df['TIPO CONTRATO'] == posicion1)]
    df_filtro2 = df[(df['PRODUCTO'] == producto2) & (df['TIPO CONTRATO'] == posicion2)]

    # Extraer el año de las posiciones seleccionadas
    year1 = int("20" + posicion1.split('/')[-1][-2:])
    year2 = int("20" + posicion2.split('/')[-1][-2:])

    # Validar que las fechas sean válidas
    def validar_fecha(mes_dia):
        try:
            return datetime.strptime(f"{year1}-{mes_dia}", '%Y-%m-%d')
        except ValueError:
            return None

    df_filtro1['FECHA'] = df_filtro1['MES-DIA'].apply(lambda x: validar_fecha(x))
    df_filtro2['FECHA'] = df_filtro2['MES-DIA'].apply(lambda x: validar_fecha(x))

    df_filtro1 = df_filtro1.dropna(subset=['FECHA'])
    df_filtro2 = df_filtro2.dropna(subset=['FECHA'])

    # Calcular el promedio por MES-DIA para cada producto
    df_promedio1 = df_filtro1.groupby('MES-DIA')['AJUSTE / PRIMA REF.'].mean().reset_index()
    df_promedio2 = df_filtro2.groupby('MES-DIA')['AJUSTE / PRIMA REF.'].mean().reset_index()

    # Convertir MES-DIA a un datetime para plotly usando el año de las posiciones seleccionadas
    df_promedio1['FECHA'] = df_promedio1['MES-DIA'].apply(lambda x: datetime.strptime(f"{year1}-{x}", '%Y-%m-%d'))
    df_promedio2['FECHA'] = df_promedio2['MES-DIA'].apply(lambda x: datetime.strptime(f"{year2}-{x}", '%Y-%m-%d'))

    # Graficar los datos
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_promedio1['FECHA'], y=df_promedio1['AJUSTE / PRIMA REF.'], mode='lines+markers', name=f'Promedio Histórico {producto1} - {posicion1}'))
    fig.add_trace(go.Scatter(x=df_promedio2['FECHA'], y=df_promedio2['AJUSTE / PRIMA REF.'], mode='lines+markers', name=f'Promedio Histórico {producto2} - {posicion2}'))

    fig.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Precio de ajuste / prima ref.',
        xaxis=dict(
            tickformat='%d/%m/%Y',
            tickmode='auto'
        ),
        legend=dict(
            yanchor="bottom",
            y=1,
            xanchor="left",
            x=0.01
        )
    )

    st.plotly_chart(fig)
