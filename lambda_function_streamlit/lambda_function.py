import json
import csv
import boto3
import requests
from datetime import datetime
from io import StringIO
import re

# Inicializar el cliente de S3
s3 = boto3.client('s3')
bucket_name = 'e-mat-spread'
csv_file_name = 'historico_comparacion.csv'

def obtener_productos_agricolas():
    try:
        # Descargar el archivo CSV existente desde S3
        existing_obj = s3.get_object(Bucket=bucket_name, Key=csv_file_name)
        existing_csv = existing_obj['Body'].read().decode('utf-8')
        csv_reader = csv.DictReader(StringIO(existing_csv))
        
        # Obtener los valores únicos de la columna PRODUCTO
        productos_agricolas = set(row['PRODUCTO'] for row in csv_reader)
        
        return productos_agricolas
    except s3.exceptions.NoSuchKey:
        return set()

def es_opcion(tipo_contrato):
    # Patrón que detecta si el contrato contiene un número seguido de una letra (p.ej. "185 C")
    patron_opcion = re.compile(r'\d+ [A-Z]')
    return bool(patron_opcion.search(tipo_contrato))

def lambda_handler(event, context):
    # Calcular la fecha de hoy
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')

    # Definir la URL del endpoint de precios de cierre
    url_closing_prices = 'https://apicem.matbarofex.com.ar/api/v2/closing-prices'

    # Definir los parámetros para la API
    params = {
        "from": today_str,
        "to": today_str,
        "market": "ROFX",
        "version": "v2"
    }

    log_messages = []  # Lista para almacenar mensajes de log

    # Obtener la lista de productos agrícolas
    productos_agricolas = obtener_productos_agricolas()

    # Realizar la solicitud a la API para obtener la lista de precios de cierre
    response_closing_prices = requests.get(url_closing_prices, params=params)
    log_messages.append(f"Estado de la respuesta de precios de cierre: {response_closing_prices.status_code}")

    # Intentar decodificar la respuesta como JSON
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

        # Contar las filas agregadas
        filas_agregadas = len(filtered_data)
        log_messages.append(f"Filas agregadas: {filas_agregadas}")

        # Descargar el archivo CSV existente desde S3
        try:
            existing_obj = s3.get_object(Bucket=bucket_name, Key=csv_file_name)
            existing_csv = existing_obj['Body'].read().decode('utf-8')
            csv_reader = csv.DictReader(StringIO(existing_csv))
            existing_data = list(csv_reader)
        except s3.exceptions.NoSuchKey:
            existing_data = []

        # Combinar los datos existentes con los nuevos datos
        combined_data = existing_data + filtered_data

        # Convertir los datos combinados a CSV
        csv_file = StringIO()
        csv_writer = csv.DictWriter(csv_file, fieldnames=["FECHA", "PRODUCTO", "TIPO CONTRATO", "AJUSTE / PRIMA REF."])
        csv_writer.writeheader()
        for row in combined_data:
            csv_writer.writerow(row)
        csv_file.seek(0)

        # Subir el archivo CSV actualizado a S3
        s3.put_object(Bucket=bucket_name, Key=csv_file_name, Body=csv_file.getvalue())
        log_messages.append(f"Datos guardados en S3: {csv_file_name}")

    except json.JSONDecodeError:
        log_messages.append("La respuesta de precios de cierre no contiene un JSON válido:")
        log_messages.append(response_closing_prices.text)

    # Guardar los mensajes de log en un archivo en S3
    log_file_name = f"logs/lambda_logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    s3.put_object(Bucket=bucket_name, Key=log_file_name, Body='\n'.join(log_messages))

    return {
        'statusCode': 200,
        'body': json.dumps(f"Proceso completado. Logs guardados en S3: {log_file_name}")
    }
