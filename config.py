import json
import os
import streamlit as st

def cargar_configuracion():

    # # Configuracion Local
    # # Cargar configuración desde el archivo config.json
    # with open("../config.json") as config_file:
    #     config = json.load(config_file)

    # # Desempaquetar las credenciales desde el archivo de configuración
    # aws_access_key = config["aws_access_key"]
    # aws_secret_key = config["aws_secret_key"]
    # region_name = config["region_name"]
    # bucket_name = config["bucket_name"]

    # # Configuracion Render
    # # Configura tus credenciales y la región de AWS desde variables de entorno
    # aws_access_key = os.getenv('aws_access_key_id')
    # aws_secret_key = os.getenv('aws_secret_access_key')
    # region_name = os.getenv('aws_region')
    # bucket_name = 'megatron-accesorios'

    #Configuracion Streamlit
    aws_access_key = st.secrets["aws_access_key"]
    aws_secret_key = st.secrets["aws_secret_key"]
    region_name = st.secrets["region_name"]
    bucket_name = st.secrets["bucket_name"]

    return aws_access_key, aws_secret_key, region_name, bucket_name
