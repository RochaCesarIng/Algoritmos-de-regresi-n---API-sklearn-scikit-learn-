import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# Cargar los datos de precios de viviendas en Melbourne
df = pd.read_csv('DatosMelbourne.csv')

# Eliminar filas con valores faltantes en la variable objetivo
df = df.dropna(subset=['Price'])
# Seleccionar características y variable objetivo
X = df[['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt']]
y = df['Price']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalizar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Crear y entrenar el modelo de Elastic Net Regression
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X_train_scaled, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = elastic_net.predict(X_test_scaled)
# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
print("Error Cuadrático Medio:", mse)
# Visualizar los coeficientes del modelo
plt.figure(figsize=(10, 5))
plt.plot(elastic_net.coef_, color='skyblue', linewidth=2, marker='o', markersize=7)
plt.title("Coeficientes del modelo Elastic Net")
plt.xlabel("Índice del coeficiente")
plt.ylabel("Valor del coeficiente")
plt.grid(True)
plt.show()
