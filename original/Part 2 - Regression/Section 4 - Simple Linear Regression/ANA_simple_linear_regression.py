#Regresón lineal simple

#Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importar dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1].values

#Dividir el data set en conjunto de entrenamiento y de testing
from sklearn.model_selection import train_test_split
X_encoded_train, X_encoded_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#Escalado de variables
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_sc_encode_train = sc_X.fit_transform(X_encoded_train)
X_sc_encode_train = sc_X.fit_transform(X_encoded_test)"""

#crear modelo de regresión con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_encoded_train, y_train)

#predecir el conjunto de test
y_pred = regression.predict(X_encoded_test)

#visualizar los datos de entrenamiento
"""plt.scatter(X_encoded_train, y_train, color ="red")
plt.plot(X_encoded_train, regression.predict(X_encoded_train), color="blue")
plt.title("Sueldo vs Años de experienia (Conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo en $")
plt.show()"""

#visualizar los datos de testing
plt.scatter(X_encoded_test, y_test, color ="red")
plt.scatter(X_encoded_test, y_pred, color="blue")
plt.title("Sueldo vs Años de experienia (Conjunto de test)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo en $")
plt.show()