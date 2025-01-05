from flask import Flask, render_template, jsonify
import threading
import paho.mqtt.client as mqtt
import csv
import os
import pickle
import numpy as np

# Flask App Initialization
app = Flask(__name__)

# Global variables to store sensor data and prediction result
sensor_data = {
    "gyro": {"roll": 0, "pitch": 0, "yaw": 0},
    "soil_moisture": 0,
    "vibration": 0
}
predictions = []  # Store predictions for display

# MQTT broker details
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPICS = {
    "gyro": "sensor/gyro",
    "soil_moisture": "sensor/soil_moisture",
    "vibration": "sensor/vibration"
}
MQTT_CLIENT_ID = "FlaskMQTTClient_005"  # Unique ID for the MQTT client

# CSV file details
CSV_FILE = "sensor_data.csv"

# Initialize the CSV file if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Update the column names
        writer.writerow(["angleX", "angleY", "angleZ", "soilMoisture", "vibration"])

# Load model using pickle
model_path = "model_longsor.pkl"  # Ensure your model file exists
with open(model_path, 'rb') as file:
    model = pickle.load(file)  # Load the model with pickle

# Function to predict from data and update predictions
def update_prediction():
    global predictions
    prediction = predict_longsor(sensor_data)
    predictions = [prediction]  # Store the latest prediction in the list

# Function to predict from data
def predict_longsor(sensor_data):
    # Extract relevant sensor data features for prediction
    features = np.array([ 
        sensor_data["soil_moisture"],
        sensor_data["vibration"],
        sensor_data["gyro"]["roll"],  # angleX
        sensor_data["gyro"]["pitch"],  # angleY
        sensor_data["gyro"]["yaw"]    # angleZ
    ]).reshape(1, -1)  # Reshape the input to be 2D as expected by the model
    
    # Make a prediction using the model
    prediction = model.predict(features)
    
    return prediction[0]  # Return the first prediction if there are multiple

# MQTT callbacks
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with code {rc}")
    for topic in MQTT_TOPICS.values():
        client.subscribe(topic)
        print(f"Subscribed to topic: {topic}")

def on_message(client, userdata, msg):
    global sensor_data
    topic = msg.topic
    payload = msg.payload.decode()
    
    # Parse the message based on the topic
    if topic == MQTT_TOPICS["gyro"]:
        sensor_data["gyro"] = eval(payload)  # Expecting {"roll": x, "pitch": y, "yaw": z}
    elif topic == MQTT_TOPICS["soil_moisture"]:
        sensor_data["soil_moisture"] = eval(payload).get("soil_moisture", 0)
    elif topic == MQTT_TOPICS["vibration"]:
        sensor_data["vibration"] = eval(payload).get("vibration", 0)
    
    print(f"Received data on topic {topic}: {payload}")
    
    # Update the prediction based on the latest sensor data
    update_prediction()

# Function to save data to a CSV file
def save_to_csv(data):
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the sensor data to the CSV
        writer.writerow([
            data["gyro"]["roll"],  # angleX
            data["gyro"]["pitch"],  # angleY
            data["gyro"]["yaw"],  # angleZ
            data["soil_moisture"],  # soilMoisture
            data["vibration"]  # vibration
        ])
    print(f"Data saved to {CSV_FILE}")

# Initialize MQTT client with a unique client_id
mqtt_client = mqtt.Client(MQTT_CLIENT_ID)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

def mqtt_loop():
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_forever()

# Flask routes
@app.route('/')
def index():
    return render_template("index.html", data=sensor_data, predictions=predictions)

@app.route('/api/data')
def get_data():
    return jsonify(sensor_data)

@app.route('/api/prediction')
def get_prediction():
    if predictions:
        prediction = predictions[-1]  # Get the latest prediction
        status = 'Risk of Landslide' if prediction > 0.5 else 'Safe'
        return jsonify({"prediction": status})
    else:
        return jsonify({"prediction": "Loading..."})


# Start Flask and MQTT clients
if __name__ == '__main__':
    # Start the MQTT loop in a separate thread
    mqtt_thread = threading.Thread(target=mqtt_loop)
    mqtt_thread.daemon = True
    mqtt_thread.start()
    
    # Run the Flask app
    app.run(debug=True)
