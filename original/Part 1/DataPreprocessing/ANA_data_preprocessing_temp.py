#Plantilla de Pre Procesado

#Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importar dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,3].values

#DIvidir el data set en conjto de entrenamiento y de testing
from sklearn.model_selection import train_test_split
X_encoded_train, X_encoded_test, y_trains, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Escalado de variables
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_sc_encode_train = sc_X.fit_transform(X_encoded_train)
X_sc_encode_train = sc_X.fit_transform(X_encoded_test)"""

