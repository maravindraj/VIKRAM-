from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import random  # Simulating sensor data (Replace with actual sensor readings)

app = Flask(__name__)

# 🔹 Define constants
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = r"C:\Users\Aravind\Downloads\capestone2 final\wildfire_detection\wildfire_detection_model_final.keras"
IMG_SIZE = (224, 224)  # Adjust if different
class_labels = ["No Fire", "Fire Detected"]

# Sensor Thresholds (Adjust based on testing)
TEMP_THRESHOLD = 50  # Example: Above 50°C → Fire risk
GAS_THRESHOLD = 300  # Example: MQ-135 value above 300 → Fire risk
SMOKE_THRESHOLD = 200  # Example: High smoke levels → Fire risk
IR_SENSOR_TRIGGERED = 1  # Example: 1 = Fire detected by IR sensor

# 🔹 Ensure `uploads/` folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 🔹 Load the trained AI model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Prevent crashes if model fails to load

# 🔹 Simulated Sensor Data Function (Replace with real sensor input)
def get_sensor_data():
    return {
        "temperature": random.uniform(30, 70),  # Simulated temperature (30°C - 70°C)
        "gas_level": random.randint(100, 400),  # Simulated gas sensor reading
        "smoke_level": random.randint(50, 300),  # Simulated smoke level
        "ir_sensor": random.choice([0, 1])  # Simulated IR sensor detection (1 = Fire)
    }

# 🔹 Homepage Route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# 🔹 Fire Detection Route (AI + Sensors)
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # 🔸 Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print(f"✅ File saved at: {file_path}")

        # 🔸 Load and preprocess image
        img = image.load_img(file_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
        print(f"🖼 Image shape: {img_array.shape}")

        # 🔸 AI Model Prediction
        prediction = model.predict(img_array)
        ai_confidence = prediction[0][0]  # Probability output
        print(f"🔮 AI Prediction Confidence: {ai_confidence}")

        # 🔸 Get Sensor Data
        sensor_data = get_sensor_data()
        temp = sensor_data["temperature"]
        gas = sensor_data["gas_level"]
        smoke = sensor_data["smoke_level"]
        ir = sensor_data["ir_sensor"]

        print(f"🌡 Temperature: {temp}°C, 🛢 Gas Level: {gas}, 💨 Smoke Level: {smoke}, 🔥 IR Sensor: {ir}")

        # 🔸 Decision Making (AI + Sensors)
        fire_detected = False  # Default: No Fire

        # 🔸 Condition 1: AI Predicts Fire (Adjust Threshold)
        if ai_confidence > 0.3:  # Lower threshold if needed
            fire_detected = True

        # 🔸 Condition 2: Sensor Data Indicates Fire
        if temp > TEMP_THRESHOLD or gas > GAS_THRESHOLD or smoke > SMOKE_THRESHOLD or ir == IR_SENSOR_TRIGGERED:
            fire_detected = True

        # 🔸 Final Classification
        final_prediction = "🔥 Fire Detected!" if fire_detected else "✅ No Fire"

        # 🔸 Return Response
        return jsonify({
            "filename": file.filename,
            "AI_Confidence": float(ai_confidence),
            "Temperature_C": temp,
            "Gas_Level": gas,
            "Smoke_Level": smoke,
            "IR_Sensor": "Detected" if ir else "Not Detected",
            "Final_Prediction": final_prediction
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
