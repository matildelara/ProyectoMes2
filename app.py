from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import os

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo y el scaler entrenados desde el directorio actual
try:
    model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler_internet.pkl')
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    app.logger.debug(f"Modelo cargado desde: {model_path}")
    app.logger.debug(f"Scaler cargado desde: {scaler_path}")
except FileNotFoundError as e:
    app.logger.error(f"Error: No se encontró 'random_forest_model.pkl' o 'scaler_internet.pkl' en {os.path.dirname(__file__)}. Asegúrate de que estén en el directorio del proyecto.")
    raise
except Exception as e:
    app.logger.error(f"Error al cargar los archivos: {str(e)}")
    raise

# Ruta para servir el formulario HTML
@app.route('/')
def home():
    return render_template('formulario.html')

# Ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        download_speed = float(request.form['download_speed'])
        router_distance = float(request.form['router_distance'])
        packet_loss_rate = float(request.form['packet_loss_rate'])
        upload_speed = float(request.form['upload_speed'])
        weather_conditions = float(request.form['weather_conditions'])
        signal_strength = float(request.form['signal_strength'])

        # Validar y ajustar signal_strength si es necesario
        if signal_strength > 0:
            app.logger.warning("Signal_strength positivo detectado. Convertido a negativo.")
            signal_strength = -abs(signal_strength)

        # Crear un DataFrame con los nombres de columnas que coincidan con el entrenamiento
        data_df = pd.DataFrame([[download_speed, router_distance, packet_loss_rate, upload_speed, weather_conditions, signal_strength]],
                             columns=['Download_speed', 'Router_distance', 'Packet_loss_rate', 'Upload_speed', 'Weather_conditions', 'Signal_strength'])
        app.logger.debug(f"DataFrame enviado: {data_df}")

        # Escalar los datos con el scaler del entrenamiento
        data_scaled = scaler.transform(data_df)

        # Realizar predicciones
        prediction = model.predict(data_scaled)
        app.logger.debug(f"Predicción de Internet Speed: {prediction[0]}")

        # Devolver las predicciones como respuesta JSON
        return jsonify({'internet_speed': round(prediction[0], 2)})
    except Exception as e:
        app.logger.error(f"Error en la predicción: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))  # Escucha en el puerto especificado por Render o 5000 por defecto