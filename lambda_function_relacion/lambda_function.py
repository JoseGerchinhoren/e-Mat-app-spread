import json
import boto3
import pandas as pd
from datetime import datetime
from io import StringIO
import itertools
import requests
import re
import csv

# Inicializar el cliente de S3
s3 = boto3.client('s3')
bucket_name = 'e-mat-spread'
csv_file_name = 'historico_ajustes_para_relacion.csv'
result_csv_name = 'historico_relaciones.csv'

def obtener_productos_agricolas(csv_content):
    # Leer el archivo CSV existente desde el contenido descargado de S3
    csv_reader = csv.DictReader(StringIO(csv_content))
    productos_agricolas = set(row['PRODUCTO'] for row in csv_reader)
    return productos_agricolas

def es_opcion(tipo_contrato):
    # Patrón que detecta si el contrato contiene un número seguido de una letra (p.ej. "185 C")
    patron_opcion = re.compile(r'\d+ [A-Z]')
    return bool(patron_opcion.search(tipo_contrato))

def lambda_handler(event, context):
    # Calcular la fecha de hoy
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')

    log_messages = []

    try:
        # Descargar el archivo CSV existente desde S3
        try:
            existing_obj = s3.get_object(Bucket=bucket_name, Key=csv_file_name)
            existing_csv = existing_obj['Body'].read().decode('utf-8')
            combined_data = pd.read_csv(StringIO(existing_csv))
        except s3.exceptions.NoSuchKey:
            combined_data = pd.DataFrame(columns=["PRODUCTO", "TIPO CONTRATO", "AJUSTE / PRIMA REF.", "AÑO", "MES-DIA"])
            log_messages.append(f"No se encontró el archivo CSV en S3: {csv_file_name}. Se ha creado un DataFrame vacío.")

        # Obtener la lista de productos agrícolas
        productos_agricolas = obtener_productos_agricolas(existing_csv)

        # Definir la URL del endpoint de precios de cierre
        url_closing_prices = 'https://apicem.matbarofex.com.ar/api/v2/closing-prices'
        params = {
            "from": today_str,
            "to": today_str,
            "market": "ROFX",
            "version": "v2"
        }

        # Realizar la solicitud a la API para obtener la lista de precios de cierre
        response_closing_prices = requests.get(url_closing_prices, params=params)
        
        if response_closing_prices.status_code != 200:
            log_messages.append(f"Error en la solicitud a la API: {response_closing_prices.status_code}")
            return {
                'statusCode': 500,
                'body': json.dumps(f"Error en la solicitud a la API: {response_closing_prices.status_code}")
            }

        try:
            data_closing_prices = response_closing_prices.json()

            # Filtrar las columnas deseadas y cambiar el formato de los nombres de las columnas
            filtered_data = [
                {
                    "FECHA": item.get("dateTime")[:10],  # Extraer solo la fecha
                    "PRODUCTO": item.get("product"),
                    "TIPO CONTRATO": item.get("symbol"),
                    "AJUSTE / PRIMA REF.": item.get("settlement")
                }
                for item in data_closing_prices.get('data', [])
                if item.get("product") in productos_agricolas and not es_opcion(item.get("symbol"))
            ]

            # Combinar los datos existentes con los nuevos datos
            new_data = pd.DataFrame(filtered_data)

            # Asegurarse de que la columna 'FECHA' esté en formato datetime
            new_data['FECHA'] = pd.to_datetime(new_data['FECHA'], format='%Y-%m-%d', errors='coerce')

            combined_data = pd.concat([combined_data, new_data]).drop_duplicates().reset_index(drop=True)

            # Asegurarse de que la columna 'FECHA' esté en formato datetime
            combined_data['FECHA'] = pd.to_datetime(combined_data['FECHA'], format='%Y-%m-%d', errors='coerce')

            # Guardar los datos combinados en S3
            combined_csv_buffer = StringIO()
            combined_data.to_csv(combined_csv_buffer, index=False)
            s3.put_object(Bucket=bucket_name, Key=csv_file_name, Body=combined_csv_buffer.getvalue())

            # Registrar la cantidad de filas guardadas en historico_ajustes_para_relacion.csv
            filas_historico_ajustes = len(new_data)
            log_messages.append(f"Filas añadidas a {csv_file_name}: {filas_historico_ajustes}")

            # Generar una lista de tuplas que contenga todas las combinaciones de productos y posiciones
            productos_posiciones = new_data[['PRODUCTO', 'TIPO CONTRATO']].drop_duplicates().values.tolist()

            # Generar todas las combinaciones de productos y posiciones, incluyendo comparaciones invertidas
            combinaciones = [(p1, t1, p2, t2) for (p1, t1), (p2, t2) in itertools.product(productos_posiciones, repeat=2)
                             if not (p1 == p2 and t1 == t2)]  # Excluir comparaciones con el mismo producto y posición

            resultados = []
            for p1, t1, p2, t2 in combinaciones:
                df_pos1 = new_data[(new_data['PRODUCTO'] == p1) & (new_data['TIPO CONTRATO'] == t1)]
                df_pos2 = new_data[(new_data['PRODUCTO'] == p2) & (new_data['TIPO CONTRATO'] == t2)]

                # Utilizar la columna 'FECHA' para hacer la combinación
                merged = pd.merge(df_pos1, df_pos2, on='FECHA', suffixes=('_pos1', '_pos2'))

                merged['SPREAD'] = (merged['AJUSTE / PRIMA REF._pos1'] - merged['AJUSTE / PRIMA REF._pos2']).round(3)
                merged['RELACION%'] = ((merged['AJUSTE / PRIMA REF._pos1'] / merged['AJUSTE / PRIMA REF._pos2'] - 1) * 100).round(3)

                for index, row in merged.iterrows():
                    resultados.append({
                        'FECHA': row['FECHA'].strftime('%Y-%m-%d'),
                        'PRODUCTO_1': p1,
                        'TIPO_CONTRATO_1': t1,
                        'PRODUCTO_2': p2,
                        'TIPO_CONTRATO_2': t2,
                        'AJUSTE_POS1': row['AJUSTE / PRIMA REF._pos1'],
                        'AJUSTE_POS2': row['AJUSTE / PRIMA REF._pos2'],
                        'SPREAD': row['SPREAD'],
                        'RELACION%': row['RELACION%']
                    })

            # Convertir los resultados a un DataFrame
            resultados_df = pd.DataFrame(resultados)

            # Convertir los nuevos datos a CSV en un buffer de memoria
            result_csv_buffer = StringIO()
            resultados_df.to_csv(result_csv_buffer, index=False, header=False)  # Eliminar header para evitar duplicados

            # Agregar las nuevas líneas al archivo existente en S3
            s3.put_object(Bucket=bucket_name, Key=result_csv_name, Body=result_csv_buffer.getvalue(), ContentType='text/csv', CacheControl='max-age=31536000')

            # Registrar la cantidad de filas guardadas en historico_relaciones.csv
            filas_spread_comparison = len(resultados_df)
            log_messages.append(f"Filas añadidas a {result_csv_name}: {filas_spread_comparison}")

        except json.JSONDecodeError:
            log_messages.append("Error al decodificar la respuesta de la API.")

    except Exception as e:
        log_messages.append(f"Error durante el procesamiento: {str(e)}")

    # Guardar los mensajes de log en un archivo en S3
    log_file_name = f"logs/lambda_relaciones_logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    s3.put_object(Bucket=bucket_name, Key=log_file_name, Body='\n'.join(log_messages))

    return {
        'statusCode': 200,
        'body': json.dumps(f"Proceso completado. Logs guardados en S3: {log_file_name}")
    }
