import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("data/user_nutritional_data.csv")
df = df.dropna()
print(f"Data loaded: {df.shape[0]} baris, {df.shape[1]} kolom")

# =========================
# 2. SPLIT FITUR & TARGET
# =========================
target = "Calories"
X = df.drop(columns=[target])
y = df[target]

# =========================
# 3. SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler disimpan.")

# =========================
# 4. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# 5. MODEL MLP (Deep Learning ringan)
# =========================
model = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),  # 3 hidden layer
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    verbose=False
)

print("Training model...")
model.fit(X_train, y_train)
print("Training selesai!")

# =========================
# 6. EVALUASI
# =========================
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\n📊 Hasil Evaluasi:")
print(f"   MAE  : {mae:.4f}")
print(f"   RMSE : {rmse:.4f}")
print(f"   R²   : {r2:.4f}")

# =========================
# 7. SIMPAN MODEL
# =========================
joblib.dump(model, "models/model_kalori_mlp.pkl")
print("\n✅ Model berhasil disimpan di models/model_kalori_mlp.pkl")