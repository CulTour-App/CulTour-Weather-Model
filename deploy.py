from flask import Flask, request, jsonify, send_file
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import requests
from datetime import datetime
import re
from keras.layers import Layer

# Flask app initialization
app = Flask(__name__)

# Custom "NotEqual" layer
class NotEqual(Layer):
    def __init__(self, **kwargs):
        super(NotEqual, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        input1, input2 = tf.split(inputs, num_or_size_splits=2, axis=-1)
        return tf.cast(tf.math.not_equal(input1, input2), tf.float32)

# Paths to model and tokenizer files
MODEL_DIR = os.getcwd()
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_model.h5")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

# Load encoder and decoder models with custom objects
encoder_model = load_model(ENCODER_PATH, custom_objects={'NotEqual': NotEqual})
decoder_model = load_model(DECODER_PATH, custom_objects={'NotEqual': NotEqual})

# Load the tokenizer
with open(TOKENIZER_PATH, 'rb') as file:
    tokenizer = pickle.load(file)

# Load latitude and longitude dataset
CSV_URL = "https://raw.githubusercontent.com/ajisakarsyi/CulTour/refs/heads/main/asset/province_latitude_longitude_data.csv"
try:
    df = pd.read_csv(CSV_URL, skiprows=1, names=["country", "province", "latitude", "longitude"])
    print("Latitude and longitude data loaded successfully.")
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    exit()

# Function to fetch weather data
def fetch_weather_data(latitude, longitude, start_date, end_date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation"
    }
    response = requests.get(url, params=params)
    return response.json()

# Function to process weather data
def process_weather_data(data):
    times = [datetime.strptime(t, "%Y-%m-%dT%H:%M") for t in data["hourly"]["time"]]
    precipitation = np.array(data["hourly"]["precipitation"])

    daily_precipitation = {}
    for time, rain in zip(times, precipitation):
        date = time.date()
        if date not in daily_precipitation:
            daily_precipitation[date] = 0
        daily_precipitation[date] += rain

    rainy_days = {date: rain > 0.1 for date, rain in daily_precipitation.items()}

    total_days = len(rainy_days)
    rainy_count = sum(rainy_days.values())
    if rainy_count / total_days > 0.5:
        weather_summary = "Mostly rainy"
    elif rainy_count / total_days > 0.25:
        weather_summary = "Partly rainy"
    else:
        weather_summary = "Mostly sunny"

    return rainy_days, weather_summary

# Function to generate weather plot
def plot_weather_with_annotations(data, rainy_days):
    times = [datetime.strptime(t, "%Y-%m-%dT%H:%M") for t in data["hourly"]["time"]]
    temperatures = data["hourly"]["temperature_2m"]
    precipitation = data["hourly"]["precipitation"]

    plt.figure(figsize=(12, 6))
    plt.plot(times, temperatures, label="Temperature (°C)", color="orange", linewidth=2)
    plt.fill_between(times, precipitation, label="Precipitation (mm)", color="blue", alpha=0.3)

    for date, is_rainy in rainy_days.items():
        if is_rainy:
            plt.axvspan(datetime.combine(date, datetime.min.time()),
                        datetime.combine(date, datetime.max.time()),
                        color="gray", alpha=0.2, label="Rainy Day" if "Rainy Day" not in plt.gca().get_legend_handles_labels()[1] else None)

    plt.title("Weather Forecast (Easy to Interpret)")
    plt.xlabel("Time")
    plt.ylabel("Weather Metrics")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "static", "weather_plot.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Function to generate chatbot response
def generate_response(input_text):
    max_input_len = encoder_model.input_shape[1]
    input_seq = pad_sequences(tokenizer.texts_to_sequences([input_text]), maxlen=max_input_len, padding='post')
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index.get("[START]", 1)

    response = []
    for _ in range(decoder_model.output_shape[1]):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1])
        sampled_word = tokenizer.index_word.get(sampled_token_index, "<OOV>")

        if sampled_word.strip() == "[END]":
            break

        if sampled_word not in {"<OOV>", "[START]", "[END]"}:
            response.append(sampled_word)

        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return ' '.join(response).capitalize()

# Flask route for weather chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    province = data.get("province", "").strip().lower()
    start_date = data.get("start_date")
    end_date = data.get("end_date")

    if not all([province, start_date, end_date]):
        return jsonify({"error": "Missing required parameters"}), 400

    # Get coordinates for the province
    province_row = df[df['province'].apply(lambda x: x.lower() == province)]
    if province_row.empty:
        return jsonify({"error": "Province not found"}), 404

    latitude = province_row.iloc[0]['latitude']
    longitude = province_row.iloc[0]['longitude']

    # Fetch weather data
    weather_data = fetch_weather_data(latitude, longitude, start_date, end_date)
    if "error" in weather_data:
        return jsonify({"error": "Failed to fetch weather data"}), 500

    # Process weather data
    rainy_days, weather_summary = process_weather_data(weather_data)

    # Generate weather plot
    plot_path = plot_weather_with_annotations(weather_data, rainy_days)

    # Generate chatbot response
    chatbot_response = generate_response(weather_summary)

    return jsonify({
        "weather_summary": weather_summary,
        "chatbot_response": chatbot_response,
        "plot_url": f"{request.url_root.rstrip('/')}/static/weather_plot.png"
    })

# Start Flask app
if __name__ == '__main__':
    app.run(debug=True)
