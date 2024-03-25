# Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# Cargar el conjunto de datos de diabetes
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de Elastic Net Regression
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

# Entrenar el modelo
elastic_net.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = elastic_net.predict(X_test)

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

