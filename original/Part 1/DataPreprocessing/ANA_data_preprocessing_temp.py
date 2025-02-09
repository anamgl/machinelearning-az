#Plantilla de Pre Procesado

#Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importar dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,3].values

#tratamiento de los NaN
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#codificar datos categóricos de la matriz X
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

#define qué columnas son categóricas (posición 0 del array) - transforma en variable dummy
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X_encoded = ct.fit_transform(X)


#codificar datos categóricos del array Y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#DIvidir el data set en conjto de entrenamiento y de testing
from sklearn.model_selection import train_test_split
X_encoded_train, X_encoded_test, y_trains, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=0)
