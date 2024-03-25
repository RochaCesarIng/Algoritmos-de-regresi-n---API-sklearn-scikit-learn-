import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Cargar el conjunto de datos de viviendas de California
california_housing = fetch_california_housing()
X, y= california_housing.data, california_housing.target

# Convertir los datos a un DataFrame de pandas
df = pd.DataFrame(data= np.c_[california_housing['data'], california_housing['target']],
                     columns= california_housing['feature_names'] + ['target'])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y ajustar el modelo Gradient Boosting Regression
gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_regressor.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = gb_regressor.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
print("Error Cuadrático Medio:", mse)

