# ==============================================================================
#                      FINAL SCRIPT: train_model.py
# ==============================================================================
# This script builds, trains, and evaluates a neural network surrogate model
# using the corrected dataset and methodology.

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import time

# --- 1. Load Data ---
print("Loading dataset...")
df = pd.read_csv('beam_deflection_dataset.csv')

# --- 2. Pre-process Data (70/15/15 Split) ---
X = df[['k0', 'k1', 'damping', 'velocity']]
y = df['w_max']

# Split into training+validation (85%) and testing (15%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
# Split the 85% block into final training (70%) and validation (15%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=(0.15/0.85), random_state=42)

# Scale input features to a [0, 1] range
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("Data pre-processing complete.")

# --- 3. Build the ANN Model ---
print("\nBuilding the model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- 4. Train the Model ---
print("\nTraining...")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, epochs=500, validation_data=(X_val_scaled, y_val),
                    batch_size=32, callbacks=[early_stopping], verbose=1)

# --- 5. Evaluate Performance ---
print("\nEvaluating model performance...")
y_pred = model.predict(X_test_scaled).flatten()
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Final R² Score: {r2:.4f}")
print(f"Final RMSE: {rmse:.4f}")

# --- 6. Visualize Results ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', lw=2)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual Deflection')
plt.ylabel('Predicted Deflection')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 7. Save the Final Model ---
print("\nSaving model...")
model.save('surrogate_model.h5')
print("Model saved successfully as 'surrogate_model.h5'")