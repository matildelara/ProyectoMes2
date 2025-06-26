from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo y el scaler entrenados
try:
    model = joblib.load("C:/Users/Laraa/Pacial2_Proyecto9/random_forest_model.pkl")
    scaler = joblib.load("C:/Users/Laraa/Pacial2_Proyecto9/scaler_internet.pkl")
    app.logger.debug("Modelo y scaler cargados correctamente.")
except FileNotFoundError:
    app.logger.error("Error: No se encontró 'random_forest_model.pkl' o 'scaler_internet.pkl'. Asegúrate de guardarlos primero en C:/Users/Laraa/Pacial2_Proyecto9.")
    exit()

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
            signal_strength = -abs(signal_strength)  # Ajuste temporal

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
    app.run(debug=True)