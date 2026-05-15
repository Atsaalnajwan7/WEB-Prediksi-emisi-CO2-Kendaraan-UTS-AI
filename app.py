# ============================================
# APLIKASI WEB PREDIKSI EMISI CO2 KENDARAAN
# UTS PRAKTIKUM KECERDASAN BUATAN
# DEPLOYMENT READY - RAILWAY (FIXED)
# ============================================

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==================== BACKPROPAGATION CLASS ====================
class BackpropagationNN:
    def __init__(self, input_size=None, hidden_size=None, output_size=None, lr=0.01):
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
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

# ==================== LOAD MODELS ====================
print("="*50)
print("Loading models for Railway Deployment...")
print("="*50)

# Mapping Fuel Type
fuel_type_map = {
    'Premium (Z)': 1,
    'Regular (X)': 0,
    'Diesel (D)': 2,
    'E85 (E)': 3
}

# Buat folder models jika belum ada
if not os.path.exists('models'):
    os.makedirs('models')
    print("📁 Folder 'models' created")

# Load models dengan error handling
models = {
    'random_forest': None,
    'decision_tree': None,
    'linear_regression': None,
    'kmeans': None,
    'backpropagation': None,
    'bp_meta': None,
    'scaler': None
}

# Coba load semua model
model_files = {
    'random_forest': 'models/random_forest.pkl',
    'decision_tree': 'models/decision_tree.pkl',
    'linear_regression': 'models/linear_regression.pkl',
    'kmeans': 'models/kmeans.pkl',
    'backpropagation': 'models/backpropagation.pkl',
    'bp_meta': 'models/bp_meta.pkl',
    'scaler': 'models/scaler.pkl'
}

for key, path in model_files.items():
    try:
        if os.path.exists(path):
            models[key] = joblib.load(path)
            print(f"✅ {key} loaded successfully from {path}")
        else:
            print(f"⚠️ {key} not found at {path}")
            models[key] = None
    except Exception as e:
        print(f"❌ Error loading {key}: {e}")
        models[key] = None

# ==================== FUNGSI PREDIKSI ====================
def predict_with_model(features_scaled, model_name):
    """Helper function untuk prediksi dengan berbagai model"""
    try:
        if model_name == 'random_forest' and models.get('random_forest') is not None:
            pred = models['random_forest'].predict(features_scaled)[0]
            return round(pred, 2)
        elif model_name == 'decision_tree' and models.get('decision_tree') is not None:
            pred = models['decision_tree'].predict(features_scaled)[0]
            return round(pred, 2)
        elif model_name == 'linear_regression' and models.get('linear_regression') is not None:
            pred = models['linear_regression'].predict(features_scaled)[0]
            return round(pred, 2)
        elif model_name == 'backpropagation' and models.get('backpropagation') is not None and models.get('bp_meta') is not None:
            bp_norm = models['backpropagation'].predict(features_scaled)[0]
            meta = models['bp_meta']
            pred_bp = bp_norm * (meta['max'] - meta['min']) + meta['min']
            return round(pred_bp, 2)
        elif model_name == 'kmeans' and models.get('kmeans') is not None:
            return int(models['kmeans'].predict(features_scaled)[0])
        else:
            return None
    except Exception as e:
        print(f"Error in {model_name} prediction: {e}")
        return None

# ==================== ROUTES ====================
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
            
            # Buat array fitur (6 fitur)
            features = np.array([[
                engine_size, cylinders, fuel_city, fuel_hwy, fuel_comb, fuel_type_num
            ]])
            
            # Standarisasi jika scaler ada
            if models.get('scaler') is not None:
                try:
                    features_scaled = models['scaler'].transform(features)
                    print("✅ Scaler applied successfully")
                except Exception as e:
                    print(f"⚠️ Scaler transform error: {e}")
                    features_scaled = features
            else:
                print("⚠️ Scaler not available, using raw features")
                features_scaled = features
            
            # Prediksi dengan semua model yang tersedia
            predictions = []
            
            # Random Forest
            pred_rf = predict_with_model(features_scaled, 'random_forest')
            if pred_rf is not None:
                predictions.append({'name': 'Random Forest ⭐', 'value': pred_rf, 'is_cluster': False, 'color': 'success'})
            
            # Decision Tree
            pred_dt = predict_with_model(features_scaled, 'decision_tree')
            if pred_dt is not None:
                predictions.append({'name': 'Decision Tree', 'value': pred_dt, 'is_cluster': False, 'color': 'primary'})
            
            # Linear Regression
            pred_lr = predict_with_model(features_scaled, 'linear_regression')
            if pred_lr is not None:
                predictions.append({'name': 'Linear Regression', 'value': pred_lr, 'is_cluster': False, 'color': 'info'})
            
            # Backpropagation
            pred_bp = predict_with_model(features_scaled, 'backpropagation')
            if pred_bp is not None:
                predictions.append({'name': 'Backpropagation', 'value': pred_bp, 'is_cluster': False, 'color': 'warning'})
            
            # K-Means (cluster)
            cluster = predict_with_model(features_scaled, 'kmeans')
            if cluster is not None:
                cluster_names = {0: 'Rendah (0-200 g/km)', 1: 'Sedang (200-300 g/km)', 2: 'Tinggi (>300 g/km)'}
                predictions.append({
                    'name': 'K-Means Clustering', 
                    'value': cluster, 
                    'cluster_name': cluster_names.get(cluster, 'Unknown'),
                    'is_cluster': True, 
                    'color': 'secondary'
                })
            
            if not predictions:
                error = "Tidak ada model yang tersedia. Pastikan file model sudah diupload."
            
        except KeyError as e:
            error = f"Form error: Field {e} tidak ditemukan"
            print(f"Form error: {e}")
        except ValueError as e:
            error = f"Input error: Pastikan semua input diisi dengan angka yang valid"
            print(f"Value error: {e}")
        except Exception as e:
            error = f"Error: {str(e)}"
            print(f"Prediction error: {error}")
    
    return render_template('predict.html', predictions=predictions, error=error)

@app.route('/comparison')
def comparison():
    """Halaman perbandingan model"""
    return render_template('comparison.html')

@app.route('/about')
def about():
    """Halaman tentang"""
    return render_template('about.html')

@app.route('/health')
def health():
    """Health check endpoint untuk Railway"""
    models_status = {
        'random_forest': models.get('random_forest') is not None,
        'decision_tree': models.get('decision_tree') is not None,
        'linear_regression': models.get('linear_regression') is not None,
        'kmeans': models.get('kmeans') is not None,
        'backpropagation': models.get('backpropagation') is not None,
        'scaler': models.get('scaler') is not None
    }
    return jsonify({
        'status': 'ok', 
        'message': 'Server is running',
        'models_loaded': models_status,
        'note': 'Jika ada model yang False, pastikan file .pkl sudah diupload'
    })

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
        
        if models.get('scaler') is not None:
            features_scaled = models['scaler'].transform(features)
        else:
            features_scaled = features
            
        if models.get('random_forest') is not None:
            prediction = models['random_forest'].predict(features_scaled)[0]
            return jsonify({'success': True, 'prediction': round(prediction, 2)})
        else:
            return jsonify({'success': False, 'error': 'Model Random Forest not loaded'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ==================== RUN APP ====================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("="*50)
    print(f"Starting Flask app on port {port}")
    print(f"Debug mode: {debug_mode}")
    print("="*50)
    
    # Tampilkan status model
    print("\n📊 Model Status:")
    for key, value in models.items():
        status = "✅ Loaded" if value is not None else "❌ Not Found"
        print(f"   {key}: {status}")
    print("="*50)
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)