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
    df_merged['SPREAD_PORCENTUAL'] = (df_merged['AJUSTE POS1'] / df_merged['AJUSTE POS2'] - 1) * 100 # Mide cuánto representa el precio del Producto 1 en relación al Producto 2, expresado como un porcentaje.

    df_merged_anterior = df_merged_anterior.rename(columns={
        'AJUSTE / PRIMA REF._pos1': 'AJUSTE POS1',
        'AJUSTE / PRIMA REF._pos2': 'AJUSTE POS2'
    })
    df_merged_anterior['SPREAD'] = df_merged_anterior['AJUSTE POS1'] - df_merged_anterior['AJUSTE POS2']

    df_merged_anterior['SPREAD_PORCENTUAL'] = (df_merged_anterior['AJUSTE POS1'] / df_merged_anterior['AJUSTE POS2'] - 1) * 100

    # Selección de años para incluir en el cálculo
    anos_disponibles = sorted(df['AÑO_CONTRATO'].dropna().unique())
    anos_seleccionados = st.multiselect('Selecciona los años a incluir en el cálculo de promedio histórico del ajuste ', options=anos_disponibles, default=anos_disponibles)

    # Crear dos columnas
    col1, col2 = st.columns(2)

    # Selección de productos en la primera columna
    with col1:
        # Selección del tipo de promedio
        tipo_promedio = st.radio('Elige cómo calcular el promedio histórico del ajuste', ['Por Semana', 'Por Día', 'Por Mes'], horizontal=True)

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

    if tipo_promedio == 'Por Día':
        # Calcular el promedio por MES-DIA para cada producto
        df_promedio1 = df_filtro1.groupby('MES-DIA')['AJUSTE / PRIMA REF.'].mean().reset_index()
        df_promedio2 = df_filtro2.groupby('MES-DIA')['AJUSTE / PRIMA REF.'].mean().reset_index()

        # Encuentra el año más frecuente en los datos filtrados
        anio_mas_frecuente1 = df_pos1['AÑO'].value_counts().idxmax()
        anio_mas_frecuente2 = df_pos2['AÑO'].value_counts().idxmax()
        # Determina el mayor de los dos años más frecuentes
        anio_mas_frecuente = max(anio_mas_frecuente1, anio_mas_frecuente2)
        
        with col2:
            anio_mas_frecuente1  = st.number_input('Año de referencia para promedio histórico del ajuste', value=anio_mas_frecuente)

        # Función para convertir MES-DIA a fecha, manejando errores de fechas inválidas
        def convertir_a_fecha(mes_dia, year):
            try:
                return datetime.strptime(f"{year}-{mes_dia}", '%Y-%m-%d')
            except ValueError:
                return None

        # Convertir MES-DIA a un datetime para plotly usando el año más frecuente
        df_promedio1['FECHA'] = df_promedio1['MES-DIA'].apply(lambda x: convertir_a_fecha(x, anio_mas_frecuente1))
        df_promedio2['FECHA'] = df_promedio2['MES-DIA'].apply(lambda x: convertir_a_fecha(x, anio_mas_frecuente1))

        # Filtrar las filas que tuvieron conversiones exitosas
        df_promedio1 = df_promedio1.dropna(subset=['FECHA'])
        df_promedio2 = df_promedio2.dropna(subset=['FECHA'])

    elif tipo_promedio == 'Por Semana':
        # Calcular el promedio por semana para cada producto
        df_filtro1['SEMANA'] = df_filtro1['FECHA'].dt.to_period('W').apply(lambda r: r.start_time)
        df_filtro2['SEMANA'] = df_filtro2['FECHA'].dt.to_period('W').apply(lambda r: r.start_time)

        df_promedio1 = df_filtro1.groupby('SEMANA')['AJUSTE / PRIMA REF.'].mean().reset_index()
        df_promedio2 = df_filtro2.groupby('SEMANA')['AJUSTE / PRIMA REF.'].mean().reset_index()

        # Encuentra el año más frecuente en los datos filtrados
        anio_mas_frecuente1 = df_pos1['AÑO'].value_counts().idxmax()
        anio_mas_frecuente2 = df_pos2['AÑO'].value_counts().idxmax()

        # Determina el mayor de los dos años más frecuentes
        anio_mas_frecuente = max(anio_mas_frecuente1, anio_mas_frecuente2)
    
        with col2:
            anio_mas_frecuente1  = st.number_input('Año de referencia para promedio histórico del ajuste', value=anio_mas_frecuente)

        # Ajustar las fechas para el promedio histórico al año más frecuente
        def ajustar_año(fecha, year):
            return fecha.replace(year=year)

        df_promedio1['FECHA'] = df_promedio1['SEMANA'].apply(lambda x: ajustar_año(x, anio_mas_frecuente1))
        df_promedio2['FECHA'] = df_promedio2['SEMANA'].apply(lambda x: ajustar_año(x, anio_mas_frecuente1))

        # Ordenar por fecha para asegurar la correcta representación
        df_promedio1 = df_promedio1.sort_values(by='FECHA').reset_index(drop=True)
        df_promedio2 = df_promedio2.sort_values(by='FECHA').reset_index(drop=True)

        # Añadir un NaN entre el final y el comienzo del siguiente ciclo para evitar que se unan
        df_promedio1 = pd.concat([df_promedio1, pd.DataFrame({'FECHA': [pd.NaT], 'AJUSTE / PRIMA REF.': [None]})], ignore_index=True)
        df_promedio2 = pd.concat([df_promedio2, pd.DataFrame({'FECHA': [pd.NaT], 'AJUSTE / PRIMA REF.': [None]})], ignore_index=True)


    else:
        # Calcular el promedio por MES para cada producto
        df_filtro1['MES'] = df_filtro1['FECHA'].dt.to_period('M')
        df_filtro2['MES'] = df_filtro2['FECHA'].dt.to_period('M')

        df_promedio1 = df_filtro1.groupby('MES')['AJUSTE / PRIMA REF.'].mean().reset_index()
        df_promedio2 = df_filtro2.groupby('MES')['AJUSTE / PRIMA REF.'].mean().reset_index()

        # Encuentra el año más frecuente en los datos filtrados
        anio_mas_frecuente1 = df_pos1['AÑO'].value_counts().idxmax()
        anio_mas_frecuente2 = df_pos2['AÑO'].value_counts().idxmax()

        # Determina el mayor de los dos años más frecuentes
        anio_mas_frecuente = max(anio_mas_frecuente1, anio_mas_frecuente2)
        
        with col2:
            anio_mas_frecuente1  = st.number_input('Año de referencia para promedio histórico del ajuste', value=anio_mas_frecuente)

        # Ajustar las fechas para el promedio histórico al año más frecuente
        def ajustar_año(fecha_periodo, year):
            return fecha_periodo.to_timestamp().replace(year=year)

        # Ajustar las fechas para el promedio histórico al año más frecuente
        df_promedio1['FECHA'] = df_promedio1['MES'].apply(lambda x: ajustar_año(x, anio_mas_frecuente1))
        df_promedio2['FECHA'] = df_promedio2['MES'].apply(lambda x: ajustar_año(x, anio_mas_frecuente1))

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
        yaxis_title='Precio de ajuste',
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
    
    # Calcular el promedio histórico
    if tipo_promedio == 'Por Día':
        df_prom1 = df_filtro1.groupby('MES-DIA')['AJUSTE / PRIMA REF.'].mean().reset_index()
        df_prom2 = df_filtro2.groupby('MES-DIA')['AJUSTE / PRIMA REF.'].mean().reset_index()
    else:
        df_filtro1['MES'] = df_filtro1['FECHA'].dt.strftime('%m')
        df_filtro2['MES'] = df_filtro2['FECHA'].dt.strftime('%m')
        df_prom1 = df_filtro1.groupby('MES')['AJUSTE / PRIMA REF.'].mean().reset_index()
        df_prom2 = df_filtro2.groupby('MES')['AJUSTE / PRIMA REF.'].mean().reset_index()

    df_prom_merged = pd.merge(df_prom1, df_prom2, on='MES-DIA' if tipo_promedio == 'Por Día' else 'MES', suffixes=('_pos1', '_pos2'))

    # Calcular el spread histórico y spread porcentual histórico
    df_prom_merged['SPREAD_PROM'] = df_prom_merged['AJUSTE / PRIMA REF._pos1'] - df_prom_merged['AJUSTE / PRIMA REF._pos2']
    df_prom_merged['SPREAD_PROM_PORCENTUAL'] = (df_prom_merged['AJUSTE / PRIMA REF._pos1'] / df_prom_merged['AJUSTE / PRIMA REF._pos2'] - 1) * 100

    # Filtrar valores NaN antes de calcular las métricas
    df_prom_merged = df_prom_merged.dropna(subset=['SPREAD_PROM', 'SPREAD_PROM_PORCENTUAL'])

    # Calcular el promedio histórico, desviación estándar y coeficiente de variación
    promedio_spread_historico = df_prom_merged['SPREAD_PROM'].mean()
    std_spread_historico = df_prom_merged['SPREAD_PROM'].std()
    cv_historico = (std_spread_historico / promedio_spread_historico * 100) if promedio_spread_historico != 0 else float('inf')

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

    # Calcular el Coeficiente de Variación (CV) en porcentaje
    cv_actual = (std_spread_actual / promedio_spread_actual * 100) if promedio_spread_actual != 0 else float('inf')
    cv_anterior = (std_spread_anterior / promedio_spread_anterior * 100) if promedio_spread_anterior != 0 else float('inf')

    # Filtrar datos para el promedio histórico en el mismo MES-DIA (o la fecha más cercana si no existe exactamente)
    df_prom_merged_hist = pd.merge(
        df_promedio1[['FECHA', 'AJUSTE / PRIMA REF.']], 
        df_promedio2[['FECHA', 'AJUSTE / PRIMA REF.']], 
        on='FECHA', 
        suffixes=('_pos1', '_pos2')
    )

    # Calcular el Spread Histórico y el Spread Porcentual Histórico
    df_prom_merged_hist['SPREAD_HISTORICO'] = df_prom_merged_hist['AJUSTE / PRIMA REF._pos1'] - df_prom_merged_hist['AJUSTE / PRIMA REF._pos2']
    df_prom_merged_hist['SPREAD_PORCENTUAL_HISTORICO'] = (df_prom_merged_hist['AJUSTE / PRIMA REF._pos1'] / df_prom_merged_hist['AJUSTE / PRIMA REF._pos2'] - 1) * 100

    # Filtrar valores NaN antes de calcular las métricas
    df_prom_merged_hist = df_prom_merged_hist.dropna(subset=['SPREAD_HISTORICO', 'SPREAD_PORCENTUAL_HISTORICO'])

    # Obtener el spread histórico más cercano a la fecha actual
    spread_historico = df_prom_merged_hist['SPREAD_HISTORICO'].iloc[-1] if not df_prom_merged_hist.empty else float('nan')
    spread_porcentual_historico = df_prom_merged_hist['SPREAD_PORCENTUAL_HISTORICO'].iloc[-1] if not df_prom_merged_hist.empty else float('nan')

    # Mostrar métricas de spread
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Spread Actual", f"{df_merged['SPREAD'].iloc[-1]:,.2f}")
        st.metric("Spread Porcentual Actual", f"{df_merged['SPREAD_PORCENTUAL'].iloc[-1]:.2f}%")
        st.metric("Promedio Spread Actual", f"{promedio_spread_actual:.2f}")
        st.metric("Desviación Estándar Spread Actual", f"{std_spread_actual:.2f}")
        st.metric("Coeficiente de Variación Actual", f"{cv_actual:.2f}%")

    with col2:
        st.metric("Spread Año Anterior", f"{df_merged_anterior['SPREAD'].iloc[-1]:,.2f}")
        st.metric("Spread Porcentual Año Anterior", f"{df_merged_anterior['SPREAD_PORCENTUAL'].iloc[-1]:.2f}%")
        st.metric("Promedio Spread Año Anterior", f"{promedio_spread_anterior:.2f}")
        st.metric("Desviación Estándar Spread Año Anterior", f"{std_spread_anterior:.2f}")
        st.metric("Coeficiente de Variación Año Anterior", f"{cv_anterior:.2f}%")

    with col3:
        # Añadir métricas del histórico
        st.metric("Spread Histórico", f"{spread_historico:.2f}")
        st.metric("Spread Porcentual Histórico", f"{spread_porcentual_historico:.2f}%")
        st.metric("Promedio Spread Histórico", f"{promedio_spread_historico:.2f}")
        st.metric("Desviación Estándar Spread Histórico", f"{std_spread_historico:.2f}")
        st.metric("Coeficiente de Variación Histórico", f"{cv_historico:.2f}%")

    with st.expander(f"Comparación de Spread"):
        # Mostrar gráfico
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_merged['FECHA'], y=df_merged['SPREAD'], mode='lines', name='Spread Actual'))
        fig.add_trace(go.Scatter(x=df_merged['FECHA'], y=df_merged['SPREAD_PORCENTUAL'], mode='lines', name='Spread Porcentual Actual', yaxis='y2'))
        fig.add_trace(go.Scatter(x=df_merged_anterior['FECHA'], y=df_merged_anterior['SPREAD'], mode='lines', name='Spread Año Anterior'))
        fig.add_trace(go.Scatter(x=df_merged_anterior['FECHA'], y=df_merged_anterior['SPREAD_PORCENTUAL'], mode='lines', name='Spread Porcentual Año Anterior', yaxis='y2'))

        fig.update_layout(title=f"Comparación de Spread entre {posicion1} y {posicion2}",
                        xaxis_title='Fecha',
                        yaxis_title='Spread (Valor)',
                        yaxis2=dict(title='Spread (Porcentaje)', overlaying='y', side='right'),
                        legend=dict(x=0, y=1.1, orientation="h"))

        st.plotly_chart(fig)

    with st.expander("Tabla de spreads", expanded=True):

        # Mostrar tabla de spreads
        st.header('Tabla de spreads:')
        
        # Ordenar las fechas de más reciente a más antigua
        df_merged = df_merged.sort_values(by='FECHA', ascending=False)

        df_merged['FECHA'] = df_merged['FECHA'].dt.strftime('%d/%m/%Y')

        # Renombrar la columna y formatear los valores como porcentaje
        df_merged = df_merged.rename(columns={'SPREAD_PORCENTUAL': 'RELACION%'})
        df_merged['RELACION%'] = df_merged['RELACION%'].apply(lambda x: f"{x:.2f}%")

        # Mostrar la tabla de spreads con las nuevas columnas, incluyendo el spread porcentual formateado
        st.dataframe(df_merged[['FECHA', 'AJUSTE POS1', 'AJUSTE POS2', 'SPREAD', 'RELACION%']])
