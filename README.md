# TP_IAA
Présenter par : NTANGA KABEYA Sephora
                MULAJA BIASUANZAMBI Rodrigue 
                MBAYA YANDA Hervé
                META NDALA Dorine 
                
#TP_1 
#DATASET :'CO2 Emissions_Canada'
# importation des bibliothèques 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Chargement des donnees  
data = pd.read_csv('/CO2 Emissions_Canada.csv')

# Sélection des variables importantes 
features = ['Engine Size(L)', 'Cylinders', 'Fuel Type', 'Transmission', 'Vehicle Class']
target = 'CO2 Emissions(g/km)'
X = data[features]
y = data[target]

# Prétraitement (encodage des variables catégorielles)
categorical = ['Fuel Type', 'Transmission', 'Vehicle Class']
numerical = ['Engine Size(L)', 'Cylinders']

preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

# Model 
model = Pipeline([
    ('preproc', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42))
])
# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Phase d'entrainnement du modèle
model.fit(X_train, y_train)

# Évaluation du modèle 
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
# Résultat
print("\n Résultat:")
print(f"MAE (Erreur Absolue Moyenne): {mae:.4f}")
print(f"MSE (Erreur Quadratique Moyenne): {mse:.4f}")
print(f"RMSE (Racine de l'Erreur Quadratique Moyenne): {rmse:.4f}")
print(f"R² (Coefficient de détermination): {r2:.2f}")

# TP_2
# DASET:'IRIS'
# importations 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D
# Chargement des données
data = load_iris()
X = data.data
y = data.target
# Encodage des labels
y_encoded = to_categorical(y)
# Séparation des données
data = load_iris()
X = data.data
y = data.target
# Reshape pour CNN 1D : (samples, timesteps, features)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_encoded, test_size=0.2, random_state=42)
# Modèle CNN
model = Sequential([
    Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(4, 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Phase d Entraînement
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)
# Évaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f'CNN Accuracy: {accuracy * 100:.2f}%')
