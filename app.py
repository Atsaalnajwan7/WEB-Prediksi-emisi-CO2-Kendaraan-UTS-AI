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

# Inisialisasi models dictionary
models = {
    'random_forest': None,
    'decision_tree': None,
    'linear_regression': None,
    'kmeans': None,
    'backpropagation': None,
    'bp_meta': None,
    'scaler': None
}

# Coba load semua model dengan try-except
print("\n📂 Checking model files...")

model_paths = {
    'random_forest': 'models/random_forest.pkl',
    'decision_tree': 'models/decision_tree.pkl',
    'linear_regression': 'models/linear_regression.pkl',
    'kmeans': 'models/kmeans.pkl',
    'backpropagation': 'models/backpropagation.pkl',
    'bp_meta': 'models/bp_meta.pkl',
    'scaler': 'models/scaler.pkl'
}

for name, path in model_paths.items():
    try:
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"✅ {name} loaded ({os.path.getsize(path)} bytes)")
        else:
            print(f"❌ {name} NOT FOUND at {path}")
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")

# Tampilkan ringkasan
print("\n" + "="*50)
print("MODELS LOADING SUMMARY:")
print("="*50)
for name, model in models.items():
    status = "✅ LOADED" if model is not None else "❌ NOT LOADED"
    print(f"  {name}: {status}")
print("="*50)

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
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
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
            
            fuel_type_num = fuel_type_map.get(fuel_type, 1)
            
            features = np.array([[
                engine_size, cylinders, fuel_city, fuel_hwy, fuel_comb, fuel_type_num
            ]])
            
            # Standarisasi
            if models.get('scaler') is not None:
                features_scaled = models['scaler'].transform(features)
            else:
                features_scaled = features
            
            # Prediksi dengan semua model
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
            
            # K-Means
            cluster = predict_with_model(features_scaled, 'kmeans')
            if cluster is not None:
                cluster_names = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
                predictions.append({
                    'name': 'K-Means Clustering', 
                    'value': cluster, 
                    'cluster_name': cluster_names.get(cluster, 'Unknown'),
                    'is_cluster': True, 
                    'color': 'secondary'
                })
            
            if not predictions:
                error = "Tidak ada model yang tersedia"
            
        except Exception as e:
            error = str(e)
            print(f"Prediction error: {error}")
    
    return render_template('predict.html', predictions=predictions, error=error)

@app.route('/comparison')
def comparison():
    return render_template('comparison.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/health')
def health():
    models_status = {k: v is not None for k, v in models.items()}
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'models_loaded': models_status
    })

@app.route('/debug-models')
def debug_models():
    """Debug endpoint untuk cek file di server"""
    import os
    files_info = {}
    
    if os.path.exists('models'):
        for f in os.listdir('models'):
            path = os.path.join('models', f)
            files_info[f] = {
                'size': os.path.getsize(path),
                'exists': True
            }
    
    return jsonify({
        'models_folder_exists': os.path.exists('models'),
        'files_in_models': files_info,
        'models_loaded': {k: v is not None for k, v in models.items()}
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("="*50)
    print(f"Starting Flask app on port {port}")
    print(f"Debug mode: {debug_mode}")
    print("="*50)
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)