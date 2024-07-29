import streamlit as st
import pandas as pd
import boto3
import io
from datetime import datetime
import plotly.graph_objects as go
import re

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
        return pd.DataFrame(columns=['AÑO', 'MES-DIA', 'PRODUCTO', 'TIPO CONTRATO', 'AJUSTE / PRIMA REF.'])

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
            return int(tipo_contrato.split('/')[-1][-2:])
        except (ValueError, IndexError):
            return None

    df['TIPO_CONTRATO_CLEAN'] = df['TIPO CONTRATO'].apply(limpiar_tipo_contrato)

    df['AÑO_CONTRATO'] = df['TIPO CONTRATO'].apply(limpiar_tipo_contrato)
    df['AÑO_CONTRATO'] = df['AÑO_CONTRATO'].astype('Int64')  # Asegúrate de que los años sean enteros
    
    # Crear columnas de fecha combinada
    df['FECHA'] = df['AÑO'].astype(str) + '-' + df['MES-DIA']
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d')

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

    # Selección de años para incluir en el cálculo
    anos_disponibles = sorted(df['AÑO_CONTRATO'].dropna().unique())
    anos_seleccionados = st.multiselect('Selecciona los años a incluir en el cálculo de promedio histórico del ajuste ', options=anos_disponibles, default=anos_disponibles)

    # Selección del tipo de promedio
    tipo_promedio = st.radio('Elige cómo calcular el promedio histórico del ajuste', ['Por Mes', 'Por Día y Mes'])

    # Función para generar el patrón de expresión regular a partir de la posición seleccionada
    def generar_patron(posicion):
        return re.compile(rf'^{posicion[:-2]}\d{{2}}$')

    # Filtrar datos por el producto, la posición seleccionada (considerando la corrección), y por los años seleccionados
    patron1 = generar_patron(posicion1)
    df_filtro1 = df[(df['PRODUCTO'] == producto1) & (df['TIPO CONTRATO'].apply(lambda x: bool(patron1.match(x)))) & (df['AÑO_CONTRATO'].isin(anos_seleccionados))]

    patron2 = generar_patron(posicion2)
    df_filtro2 = df[(df['PRODUCTO'] == producto2) & (df['TIPO CONTRATO'].apply(lambda x: bool(patron2.match(x)))) & (df['AÑO_CONTRATO'].isin(anos_seleccionados))]

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

    if tipo_promedio == 'Por Día y Mes':
        # Calcular el promedio por MES-DIA para cada producto
        df_promedio1 = df_filtro1.groupby('MES-DIA')['AJUSTE / PRIMA REF.'].mean().reset_index()
        df_promedio2 = df_filtro2.groupby('MES-DIA')['AJUSTE / PRIMA REF.'].mean().reset_index()

        # Convertir MES-DIA a un datetime para plotly usando el año de las posiciones seleccionadas
        df_promedio1['FECHA'] = df_promedio1['MES-DIA'].apply(lambda x: datetime.strptime(f"{year1}-{x}", '%Y-%m-%d'))
        df_promedio2['FECHA'] = df_promedio2['MES-DIA'].apply(lambda x: datetime.strptime(f"{year2}-{x}", '%Y-%m-%d'))
    else:
        # Calcular el promedio por MES para cada producto
        df_filtro1['MES'] = df_filtro1['FECHA'].dt.to_period('M')
        df_filtro2['MES'] = df_filtro2['FECHA'].dt.to_period('M')

        df_promedio1 = df_filtro1.groupby('MES')['AJUSTE / PRIMA REF.'].mean().reset_index()
        df_promedio2 = df_filtro2.groupby('MES')['AJUSTE / PRIMA REF.'].mean().reset_index()

        # Convertir MES a un datetime para plotly
        df_promedio1['FECHA'] = df_promedio1['MES'].dt.to_timestamp()
        df_promedio2['FECHA'] = df_promedio2['MES'].dt.to_timestamp()

    # Crear gráfico interactivo con Plotly
    fig = go.Figure()

    # Añadir trazas para los datos de los productos y posiciones seleccionadas
    fig.add_trace(go.Scatter(x=df_merged['FECHA'], y=df_merged['AJUSTE POS1'], 
                            mode='lines+markers', name=f'{producto1} {posicion1}', 
                            line=dict(color='orangered'), marker=dict(color='orangered')))

    fig.add_trace(go.Scatter(x=df_merged['FECHA'], y=df_merged['AJUSTE POS2'], 
                            mode='lines+markers', name=f'{producto2} {posicion2}', 
                            line=dict(color='dodgerblue'), marker=dict(color='dodgerblue')))

    # Añadir líneas del año anterior con estilo diferente
    fig.add_trace(go.Scatter(x=df_merged_anterior['FECHA'], y=df_merged_anterior['AJUSTE POS1'], 
                            mode='lines', line=dict(color='orangered', dash='dot'), 
                            name=f'{producto1} {posicion1_anterior}'))

    fig.add_trace(go.Scatter(x=df_merged_anterior['FECHA'], y=df_merged_anterior['AJUSTE POS2'], 
                            mode='lines', line=dict(color='dodgerblue', dash='dot'), 
                            name=f'{producto2} {posicion2_anterior}'))

    # Añadir trazas para el promedio histórico
    fig.add_trace(go.Scatter(x=df_promedio1['FECHA'], y=df_promedio1['AJUSTE / PRIMA REF.'], 
                            mode='lines+markers', name=f'Promedio Histórico {producto1} - {posicion1}',
                            line=dict(color='darkorange', dash='dot'), opacity=0.6))

    fig.add_trace(go.Scatter(x=df_promedio2['FECHA'], y=df_promedio2['AJUSTE / PRIMA REF.'], 
                            mode='lines+markers', name=f'Promedio Histórico {producto2} - {posicion2}', 
                            line=dict(color='aqua', dash='dot'), opacity=0.6))

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

    # Asegurarse de que la fecha completa se muestre en los puntos
    fig.update_traces(
        hovertemplate='Fecha: %{x|%d/%m/%Y}<br>Precio: %{y}'
    )

    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)

    # Calcular promedio y desviación estándar de spread de los instrumentos seleccionados
    promedio_spread_actual = df_merged['SPREAD'].mean()
    std_spread_actual = df_merged['SPREAD'].std()
    promedio_spread_anterior = df_merged_anterior['SPREAD'].mean()
    std_spread_anterior = df_merged_anterior['SPREAD'].std()

    # Spread actual
    spread_actual = df_merged['SPREAD'].iloc[-1]

    # Calcular el umbral para recomendaciones basadas en desvíos estándar del promedio histórico
    umbral_bajo = promedio_spread_actual - std_spread_actual
    umbral_alto = promedio_spread_actual + std_spread_actual

    # Recomendación basada en el spread
    st.header('Recomendación:')

    if spread_actual > umbral_alto:
        st.write(f'El spread más reciente ({spread_actual:.1f}) es significativamente mayor que el promedio histórico ({promedio_spread_actual:.2f}).')
        st.write(f'Recomendación: Vender {posicion1} y comprar {posicion2}.')
        st.write('Nivel de recomendación: Alto')
    elif spread_actual < umbral_bajo:
        st.write(f'El spread más reciente ({spread_actual:.1f}) es significativamente menor que el promedio histórico ({promedio_spread_actual:.2f}).')
        st.write(f'Recomendación: Comprar {posicion1} y vender {posicion2}.')
        st.write('Nivel de recomendación: Alto')
    else:
        if spread_actual > promedio_spread_actual:
            st.write(f'El spread más reciente ({spread_actual:.1f}) es poco mayor que el promedio histórico ({promedio_spread_actual:.2f}).')
            st.write(f'Recomendación: Considerar vender {posicion1} y comprar {posicion2}.')
            st.write('Nivel de recomendación: Baja')
        elif spread_actual < promedio_spread_actual:
            st.write(f'El spread más reciente ({spread_actual:.1f}) es poco menor que el promedio histórico ({promedio_spread_actual:.2f}).')
            st.write(f'Recomendación: Considerar comprar {posicion1} y vender {posicion2}.')
            st.write('Nivel de recomendación: Baja')
        else:
            st.write(f'El spread más reciente ({spread_actual:.1f}) es igual al promedio histórico ({promedio_spread_actual:.2f}).')
            st.write('Recomendación: No hay una acción específica recomendada.')
            st.write('Nivel de recomendación: Neutra')

    # Mostrar el promedio y la desviación estándar de spread de los instrumentos seleccionados
    st.header(f'Promedio del Spread: {promedio_spread_actual:.2f}')
    st.write(f'Promedio de Spread del Año Anterior: {promedio_spread_anterior:.2f}')

    st.header(f'Desviación Estándar del Spread: {std_spread_actual:.2f}')
    st.write(f'Desviación Estándar del Spread del Año Anterior: {std_spread_anterior:.2f}')

    # Calcular el Coeficiente de Variación (CV) en porcentaje
    cv_actual = (std_spread_actual / promedio_spread_actual * 100) if promedio_spread_actual != 0 else float('inf')
    cv_anterior = (std_spread_anterior / promedio_spread_anterior * 100) if promedio_spread_anterior != 0 else float('inf')

    # Mostrar el CV en porcentaje
    st.header(f'Coeficiente de Variación del Spread: {cv_actual:.2f}%')
    st.write(f'Coeficiente de Variación del Spread del Año Anterior: {cv_anterior:.2f}%')

    # Mostrar tabla de spreads
    st.header('Tabla de spreads:')
    
    # Ordenar las fechas de más reciente a más antigua
    df_merged = df_merged.sort_values(by='FECHA', ascending=False)

    df_merged['FECHA'] = df_merged['FECHA'].dt.strftime('%d/%m/%Y')

    # Mostrar la tabla de spreads con las nuevas columnas
    st.dataframe(df_merged[['FECHA', 'AJUSTE POS1', 'AJUSTE POS2', 'SPREAD']])