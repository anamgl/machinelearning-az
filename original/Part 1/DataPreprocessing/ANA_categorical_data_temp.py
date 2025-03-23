#Plantilla de Pre Procesado -datos

#Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importar dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,3].values

#codificar datos categóricos de la matriz X
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

#define qué columnas son categóricas (posición 0 del array) - transforma en variable dummy
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X= ct.fit_transform(X)


#codificar datos categóricos del array Y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)