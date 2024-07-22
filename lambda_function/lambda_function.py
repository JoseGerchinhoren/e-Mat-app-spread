import json
import csv
import boto3
import requests
from datetime import datetime

# Inicializar el cliente de S3
s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Definir la URL del endpoint de precios de cierre
    url_closing_prices = 'https://apicem.matbarofex.com.ar/api/v2/closing-prices'

    # Definir los parámetros para la API
    params = {
        "from": "2024-07-19",
        "to": "2024-07-19",
        "market": "ROFX",
        "version": "v2"
    }

    log_messages = []  # Lista para almacenar mensajes de log

    # Realizar la solicitud a la API para obtener la lista de precios de cierre
    response_closing_prices = requests.get(url_closing_prices, params=params)
    log_messages.append(f"Estado de la respuesta de precios de cierre: {response_closing_prices.status_code}")

    # Intentar decodificar la respuesta como JSON
    try:
        data_closing_prices = response_closing_prices.json()

        # Filtrar las columnas deseadas y cambiar el formato de los nombres de las columnas
        filtered_data = [
            {
                "PRODUCTO": item.get("product"),
                "TIPO CONTRATO": item.get("symbol"),
                "FECHA": item.get("dateTime")[:10],  # Extraer solo la fecha
                "AJUSTE / PRIMA REF.": item.get("settlement")
            }
            for item in data_closing_prices.get('data', [])
        ]

        # Convertir los datos a CSV
        from io import StringIO
        csv_file = StringIO()
        csv_writer = csv.DictWriter(csv_file, fieldnames=["PRODUCTO", "TIPO CONTRATO", "FECHA", "AJUSTE / PRIMA REF."])
        csv_writer.writeheader()
        for row in filtered_data:
            csv_writer.writerow(row)
        csv_file.seek(0)

        # Nombre del archivo CSV en S3
        csv_file_name = 'historico_spread.csv'

        # Subir el archivo CSV a S3
        s3.put_object(Bucket='e-mat-spread', Key=csv_file_name, Body=csv_file.getvalue())
        log_messages.append(f"Datos guardados en S3: {csv_file_name}")

    except json.JSONDecodeError:
        log_messages.append("La respuesta de precios de cierre no contiene un JSON válido:")
        log_messages.append(response_closing_prices.text)

    # Guardar los mensajes de log en un archivo en S3
    log_file_name = f"logs/lambda_logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    s3.put_object(Bucket='e-mat-spread', Key=log_file_name, Body='\n'.join(log_messages))

    return {
        'statusCode': 200,
        'body': json.dumps(f"Proceso completado. Logs guardados en S3: {log_file_name}")
    }
