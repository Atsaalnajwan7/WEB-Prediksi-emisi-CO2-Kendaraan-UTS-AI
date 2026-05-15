# ============================================
# APLIKASI WEB PREDIKSI EMISI CO2 KENDARAAN
# UTS PRAKTIKUM KECERDASAN BUATAN
# ============================================

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Class Backpropagation untuk unpickling
class BackpropagationNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        self.lr = lr
        self.losses = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def predict(self, X):
        return self.forward(X).flatten()

# Load semua model dan scaler
rf_model = joblib.load('models/random_forest.pkl')
dt_model = joblib.load('models/decision_tree.pkl')
lr_model = joblib.load('models/linear_regression.pkl')
kmeans_model = joblib.load('models/kmeans.pkl')
bp_model = joblib.load('models/backpropagation.pkl')
bp_meta = joblib.load('models/bp_meta.pkl')
scaler = joblib.load('models/scaler.pkl')

# Mapping Fuel Type
fuel_type_map = {
    'Premium (Z)': 1,
    'Regular (X)': 0,
    'Diesel (D)': 2,
    'E85 (E)': 3
}

@app.route('/')
def home():
    """Halaman utama"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Halaman prediksi"""
    predictions = None
    error = None
    
    if request.method == 'POST':
        try:
            # Ambil data dari form
            engine_size = float(request.form['engine_size'])
            cylinders = int(request.form['cylinders'])
            fuel_city = float(request.form['fuel_city'])
            fuel_hwy = float(request.form['fuel_hwy'])
            fuel_comb = float(request.form['fuel_comb'])
            fuel_type = request.form['fuel_type']
            
            # Konversi fuel type ke numeric
            fuel_type_num = fuel_type_map.get(fuel_type, 1)
            
            # Buat array fitur
            features = np.array([[engine_size, cylinders, fuel_city, fuel_hwy, fuel_comb, fuel_type_num]])
            
            # Standarisasi
            features_scaled = scaler.transform(features)
            
            # Prediksi dengan semua model
            pred_rf = round(rf_model.predict(features_scaled)[0], 2)
            pred_dt = round(dt_model.predict(features_scaled)[0], 2)
            pred_lr = round(lr_model.predict(features_scaled)[0], 2)
            
            # K-Means memprediksi cluster
            cluster = kmeans_model.predict(features_scaled)[0]
            
            # Backpropagation (perlu denormalisasi)
            bp_norm = bp_model.predict(features_scaled)[0]
            pred_bp = bp_norm * (bp_meta['max'] - bp_meta['min']) + bp_meta['min']
            pred_bp = round(pred_bp, 2)
            
            predictions = [
                {'name': 'Random Forest', 'value': pred_rf, 'is_cluster': False},
                {'name': 'Decision Tree', 'value': pred_dt, 'is_cluster': False},
                {'name': 'Linear Regression', 'value': pred_lr, 'is_cluster': False},
                {'name': 'Backpropagation', 'value': pred_bp, 'is_cluster': False},
                {'name': 'K-Means Clustering', 'value': int(cluster), 'is_cluster': True}
            ]
            
        except Exception as e:
            error = str(e)
    
    return render_template('predict.html', predictions=predictions, error=error)



@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint untuk prediksi"""
    try:
        data = request.get_json()
        features = np.array([[
            data['engine_size'],
            data['cylinders'],
            data['fuel_city'],
            data['fuel_hwy'],
            data['fuel_comb'],
            fuel_type_map.get(data['fuel_type'], 1)
        ]])
        features_scaled = scaler.transform(features)
        prediction = rf_model.predict(features_scaled)[0]
        return jsonify({'success': True, 'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)