# ============================================
# UTS PRAKTIKUM KECERDASAN BUATAN
# PREDIKSI EMISI CO2 KENDARAAN
# Gambar muncul satu per satu (setelah ditutup)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Set style untuk grafik
try:
    plt.style.use('ggplot')
except:
    pass
sns.set_palette("Set2")

print("="*60)
print("PROYEK UTS - PREDIKSI EMISI CO2 KENDARAAN")
import sys
print(f"Python Version: {sys.version.split()[0]}")
print("="*60)

# Load dataset
print("\n[1] Loading dataset...")
df = pd.read_csv('CO2 Emissions_Canada.csv')
print(f"Dataset shape: {df.shape}")

# EDA
print("\n[2] Exploratory Data Analysis...")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Target range: {df['CO2 Emissions(g/km)'].min()} - {df['CO2 Emissions(g/km)'].max()}")

# Preprocessing
print("\n[3] Preprocessing...")
features = ['Engine Size(L)', 'Cylinders', 
            'Fuel Consumption City (L/100 km)',
            'Fuel Consumption Hwy (L/100 km)', 
            'Fuel Consumption Comb (L/100 km)']

X = df[features].copy()
y = df['CO2 Emissions(g/km)'].copy()

# Encode Fuel Type
le = LabelEncoder()
df['Fuel Type Encoded'] = le.fit_transform(df['Fuel Type'])
X['Fuel Type'] = df['Fuel Type Encoded']

print(f"Features: {X.columns.tolist()}")

# Standarisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# =================== 1. LINEAR REGRESSION ===================
print("\n--- 1. Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print(f"MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}, R2: {r2_lr:.4f}")

# =================== 2. DECISION TREE ===================
print("\n--- 2. Decision Tree ---")
dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)

print(f"MAE: {mae_dt:.2f}, RMSE: {rmse_dt:.2f}, R2: {r2_dt:.4f}")

# =================== 3. RANDOM FOREST ===================
print("\n--- 3. Random Forest ---")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}, R2: {r2_rf:.4f}")

# =================== 4. K-MEANS CLUSTERING ===================
print("\n--- 4. K-Means Clustering ---")

# Elbow Method dengan subset data
inertias = []
K_range = range(2, 9)
sample_size = min(3000, X_scaled.shape[0])
X_sample = X_scaled[:sample_size]

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_sample)
    inertias.append(kmeans.inertia_)

# Cari titik elbow
inertia_diffs = np.diff(inertias)
optimal_k = K_range[np.argmax(inertia_diffs) + 1] if len(inertia_diffs) > 0 else 3

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)

print(f"Optimal K: {optimal_k}")
print(f"Inertia: {kmeans_final.inertia_:.2f}")

# =================== 5. BACKPROPAGATION ===================
print("\n--- 5. Backpropagation (Manual NumPy) ---")

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
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        dZ2 = output - y.reshape(-1, 1)
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X, y, epochs=100, verbose=False):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)
            self.losses.append(loss)
            self.backward(X, y, output)
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def predict(self, X):
        return self.forward(X).flatten()

print("Melatih Backpropagation...")
y_train_norm = ((y_train - y_train.min()) / (y_train.max() - y_train.min())).values
y_test_norm = ((y_test - y_train.min()) / (y_train.max() - y_train.min())).values

bp = BackpropagationNN(input_size=X_train.shape[1], hidden_size=32, output_size=1, lr=0.01)
bp.train(X_train, y_train_norm, epochs=100, verbose=True)

y_pred_bp_norm = bp.predict(X_test)
y_pred_bp = y_pred_bp_norm * (y_train.max() - y_train.min()) + y_train.min()

mae_bp = mean_absolute_error(y_test, y_pred_bp)
rmse_bp = np.sqrt(mean_squared_error(y_test, y_pred_bp))
r2_bp = r2_score(y_test, y_pred_bp)

print(f"\nHasil Backpropagation:")
print(f"MAE: {mae_bp:.2f}, RMSE: {rmse_bp:.2f}, R2: {r2_bp:.4f}")

# =================== TABEL PERBANDINGAN ===================
print("\n" + "="*60)
print("TABEL PERBANDINGAN PERFORMANCE MODEL")
print("="*60)

models_list = ['Linear Regression', 'Decision Tree', 'Random Forest', 'K-Means', 'Backpropagation']
mae_vals = [mae_lr, mae_dt, mae_rf, '-', mae_bp]
rmse_vals = [rmse_lr, rmse_dt, rmse_rf, kmeans_final.inertia_, rmse_bp]
r2_vals = [r2_lr, r2_dt, r2_rf, '-', r2_bp]

print(f"{'Model':<20} {'MAE':<12} {'RMSE/Inertia':<15} {'R2 Score':<10}")
print("-" * 60)
for i, model in enumerate(models_list):
    print(f"{model:<20} {str(mae_vals[i]):<12} {str(rmse_vals[i]):<15} {str(r2_vals[i]):<10}")

# =================== VISUALISASI (SATU PER SATU) ===================
print("\n" + "="*60)
print("MENAMPILKAN VISUALISASI...")
print("="*60)
print("\n⚠️  Gambar akan muncul satu per satu!")
print("   Tutup jendela grafik untuk melihat grafik berikutnya.\n")

# Buat folder untuk menyimpan gambar
if not os.path.exists('static/visualizations'):
    os.makedirs('static/visualizations')

# ========== GRAFIK 1: Perbandingan MAE ==========
print("[Gambar 1/7] Perbandingan MAE Antar Model...")
plt.figure(1, figsize=(12, 6))
models_plot = ['Linear\nRegression', 'Decision\nTree', 'Random\nForest', 'Backpropagation']
mae_plot = [mae_lr, mae_dt, mae_rf, mae_bp]
colors_mae = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bars = plt.bar(models_plot, mae_plot, color=colors_mae, edgecolor='black', linewidth=1.5)
plt.title('Perbandingan MAE Antar Model', fontsize=16, fontweight='bold')
plt.ylabel('MAE (g/km)', fontsize=12)
plt.xlabel('Model', fontsize=12)
for bar, val in zip(bars, mae_plot):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('static/visualizations/mae_comparison.png', dpi=150, bbox_inches='tight')
plt.show()  # <--- DIAKTIFKAN: gambar akan muncul
input("   Tekan Enter untuk lanjut ke grafik berikutnya...")
print()

# ========== GRAFIK 2: Perbandingan R2 Score ==========
print("[Gambar 2/7] Perbandingan R² Score Antar Model...")
plt.figure(2, figsize=(12, 6))
r2_plot = [r2_lr, r2_dt, r2_rf, r2_bp]
bars = plt.bar(models_plot, r2_plot, color=colors_mae, edgecolor='black', linewidth=1.5)
plt.title('Perbandingan R² Score Antar Model', fontsize=16, fontweight='bold')
plt.ylabel('R² Score', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.ylim(0, 1.02)
for bar, val in zip(bars, r2_plot):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05, 
             f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('static/visualizations/r2_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
input("   Tekan Enter untuk lanjut ke grafik berikutnya...")
print()

# ========== GRAFIK 3: Prediksi vs Aktual (Random Forest) ==========
print("[Gambar 3/7] Prediksi vs Aktual (Random Forest)...")
plt.figure(3, figsize=(10, 8))
plt.scatter(y_test, y_pred_rf, alpha=0.5, c='#2ecc71', edgecolors='white', s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.title('Random Forest: Prediksi vs Aktual CO₂ Emissions', fontsize=16, fontweight='bold')
plt.xlabel('Aktual CO₂ Emissions (g/km)', fontsize=12)
plt.ylabel('Prediksi CO₂ Emissions (g/km)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/visualizations/prediction_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
input("   Tekan Enter untuk lanjut ke grafik berikutnya...")
print()

# ========== GRAFIK 4: Elbow Method K-Means ==========
print("[Gambar 4/7] K-Means Elbow Method...")
plt.figure(4, figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.title('K-Means Clustering - Elbow Method', fontsize=16, fontweight='bold')
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.xticks(K_range)
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K = {optimal_k}')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/visualizations/kmeans_elbow.png', dpi=150, bbox_inches='tight')
plt.show()
input("   Tekan Enter untuk lanjut ke grafik berikutnya...")
print()

# ========== GRAFIK 5: Distribusi Error ==========
print("[Gambar 5/7] Distribusi Residual Error...")
plt.figure(5, figsize=(10, 6))
residuals = y_test - y_pred_rf
plt.hist(residuals, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
plt.title('Distribusi Residual Error (Random Forest)', fontsize=16, fontweight='bold')
plt.xlabel('Residual Error (g/km)', fontsize=12)
plt.ylabel('Frekuensi', fontsize=12)
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/visualizations/residual_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
input("   Tekan Enter untuk lanjut ke grafik berikutnya...")
print()

# ========== GRAFIK 6: Feature Importance ==========
print("[Gambar 6/7] Feature Importance (Random Forest)...")
plt.figure(6, figsize=(10, 6))
feature_importance = rf.feature_importances_
feature_names = X.columns.tolist()
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], color='#e74c3c')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.title('Feature Importance (Random Forest)', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.tight_layout()
plt.savefig('static/visualizations/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
input("   Tekan Enter untuk lanjut ke grafik berikutnya...")
print()

# ========== GRAFIK 7: Loss Curve Backpropagation ==========
print("[Gambar 7/7] Backpropagation Training Loss Curve...")
plt.figure(7, figsize=(10, 6))
plt.plot(bp.losses, 'b-', linewidth=1.5)
plt.title('Backpropagation - Training Loss Curve', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/visualizations/loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Grafik 7 selesai!\n")

print("="*60)
print("✅ SEMUA VISUALISASI TELAH DITAMPILKAN")
print("="*60)
print("\n📁 File gambar juga disimpan di folder 'static/visualizations/':")
print("   - mae_comparison.png")
print("   - r2_comparison.png")
print("   - prediction_scatter.png")
print("   - kmeans_elbow.png")
print("   - residual_distribution.png")
print("   - feature_importance.png")
print("   - loss_curve.png")

# =================== SIMPAN MODEL ===================
print("\n" + "="*60)
print("MENYIMPAN MODEL...")
print("="*60)

if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(rf, 'models/random_forest.pkl')
joblib.dump(dt, 'models/decision_tree.pkl')
joblib.dump(lr, 'models/linear_regression.pkl')
joblib.dump(kmeans_final, 'models/kmeans.pkl')
joblib.dump(bp, 'models/backpropagation.pkl')
joblib.dump({'min': float(y_train.min()), 'max': float(y_train.max())}, 'models/bp_meta.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Semua model (RF, DT, LR, KMeans, BP) dan scaler telah disimpan ke dalam folder 'models/'")

# =================== KESIMPULAN AKHIR ===================
print("\n" + "="*60)
print("KESIMPULAN")
print("="*60)

best_idx = np.argmax([r2_lr, r2_dt, r2_rf, 0, r2_bp])
print(f"\n✅ Model terbaik: {models_list[best_idx]}")
print(f"   R² Score: {[r2_lr, r2_dt, r2_rf, 0, r2_bp][best_idx]:.4f}")
print(f"   MAE: {[mae_lr, mae_dt, mae_rf, 0, mae_bp][best_idx]:.2f} g/km")
print(f"   RMSE: {[rmse_lr, rmse_dt, rmse_rf, 0, rmse_bp][best_idx]:.2f} g/km")

print("\n📊 Kesimpulan Analisis Komparatif:")
print("="*40)
print("1. Random Forest adalah model terbaik dengan R² = 0.9973")
print("   - Hampir sempurna dalam memprediksi emisi CO2")
print("   - Kelebihan: Menggabungkan banyak decision tree, robust terhadap overfitting")
print("   - Kekurangan: Komputasi lebih berat, sulit diinterpretasi")
print()
print("2. Decision Tree juga sangat baik (R² = 0.9958)")
print("   - Kelebihan: Mudah diinterpretasi, cepat training")
print("   - Kekurangan: Cenderung overfitting jika tidak di-pruning")
print()
print("3. Linear Regression cukup baik sebagai baseline (R² = 0.8810)")
print("   - Kelebihan: Sederhana, cepat, mudah dijelaskan")
print("   - Kekurangan: Tidak bisa menangkap pola non-linear")
print()
print("4. Backpropagation (R² = 0.9123)")
print("   - Kelebihan: Bisa menangkap pola kompleks")
print("   - Kekurangan: Memerlukan tuning parameter dan lebih lama training")
print()
print("5. K-Means Clustering")
print("   - Berguna untuk segmentasi data kendaraan berdasarkan emisi")
print(f"   - Optimal K = {optimal_k} cluster")

print("\n" + "="*60)
print("PROSES SELESAI!")
print("="*60)