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
        return pd.DataFrame(columns=['FECHA', 'PRODUCTO', 'TIPO CONTRATO', 'AJUSTE / PRIMA REF.'])

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
    df['FECHA'] = pd.to_datetime(df['FECHA'])

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

    def get_previous_year_position(position):
        # Obtener el año de la posición seleccionada
        year = int(position[-2:]) + 2000
        # Generar la posición del año anterior
        previous_year = str(year - 1)[-2:]
        previous_position = position[:-2] + previous_year
        return previous_position

    def adjust_date_to_next_year(date):
        try:
            return date.replace(year=date.year + 1)
        except ValueError:
            # Ajuste para fechas como 29 de febrero en años no bisiestos
            return date.replace(year=date.year + 1, month=3, day=1)

    # Filtrar datos para los productos y posiciones seleccionados
    df_pos1 = df[(df['PRODUCTO'] == producto1) & (df['TIPO CONTRATO'] == posicion1)]
    df_pos2 = df[(df['PRODUCTO'] == producto2) & (df['TIPO CONTRATO'] == posicion2)]

    # Filtrar datos para los productos y posiciones del año anterior
    posicion1_anterior = get_previous_year_position(posicion1)
    posicion2_anterior = get_previous_year_position(posicion2)
    df_pos1_anterior = df[(df['PRODUCTO'] == producto1) & (df['TIPO CONTRATO'] == posicion1_anterior)]
    df_pos2_anterior = df[(df['PRODUCTO'] == producto2) & (df['TIPO CONTRATO'] == posicion2_anterior)]

    # Ajustar las fechas del año anterior para que coincidan con el año actual
    df_pos1_anterior['FECHA'] = df_pos1_anterior['FECHA'].apply(adjust_date_to_next_year)
    df_pos2_anterior['FECHA'] = df_pos2_anterior['FECHA'].apply(adjust_date_to_next_year)

    # Unir dataframes por fecha
    df_merged = pd.merge(df_pos1, df_pos2, on='FECHA', suffixes=('_pos1', '_pos2'))
    df_merged_anterior = pd.merge(df_pos1_anterior, df_pos2_anterior, on='FECHA', suffixes=('_pos1', '_pos2'))

    # Renombrar columnas
    df_merged = df_merged.rename(columns={
        'AJUSTE / PRIMA REF._pos1': 'AJUSTE POS1',
        'AJUSTE / PRIMA REF._pos2': 'AJUSTE POS2'
    })

    # Calcular spread después de renombrar las columnas
    df_merged['SPREAD'] = df_merged['AJUSTE POS1'] - df_merged['AJUSTE POS2']
    df_merged_anterior = df_merged_anterior.rename(columns={
        'AJUSTE / PRIMA REF._pos1': 'AJUSTE POS1',
        'AJUSTE / PRIMA REF._pos2': 'AJUSTE POS2'
    })
    df_merged_anterior['SPREAD'] = df_merged_anterior['AJUSTE POS1'] - df_merged_anterior['AJUSTE POS2']

    # Crear gráfico interactivo con Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_merged['FECHA'], y=df_merged['AJUSTE POS1'], 
                             mode='lines+markers', name=f'{producto1} {posicion1}'))

    fig.add_trace(go.Scatter(x=df_merged['FECHA'], y=df_merged['AJUSTE POS2'], 
                             mode='lines+markers', name=f'{producto2} {posicion2}'))

    # Añadir líneas del año anterior con estilo diferente
    fig.add_trace(go.Scatter(x=df_merged_anterior['FECHA'], y=df_merged_anterior['AJUSTE POS1'], 
                             mode='lines', line=dict(dash='dot'), name=f'{producto1} {posicion1_anterior}'))

    fig.add_trace(go.Scatter(x=df_merged_anterior['FECHA'], y=df_merged_anterior['AJUSTE POS2'], 
                             mode='lines', line=dict(dash='dot'), name=f'{producto2} {posicion2_anterior}'))

    fig.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Precio de ajuste / prima ref.',
        xaxis=dict(
            tickformat='%d/%m/%Y',
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
    
    # Ordenar las fechas de más reciente a más antigua
    df_merged = df_merged.sort_values(by='FECHA', ascending=False)

    df_merged['FECHA'] = df_merged['FECHA'].dt.strftime('%d/%m/%Y')

    # Mostrar la tabla de spreads con las nuevas columnas
    st.write(df_merged[['FECHA', 'AJUSTE POS1', 'AJUSTE POS2', 'SPREAD']])
