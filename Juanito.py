import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# clasificacion

iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Clasificar las flores como "margarita" o "no margarita"
df['clasificacion'] = df.apply(lambda row: 'margarita' if row['sepal length (cm)'] >= 5.1 and row['sepal width (cm)'] >=3.5 and row['petal length (cm)'] >=1.3 and row['petal width (cm)'] <=0.2 else 'no margarita', axis=1)

# Mostrar los datos que son margarita
print("Datos que son margarita:")
print(df[df['clasificacion'] == 'margarita'])

# Mostrar los datos que no son margarita
print("Datos que no son margarita:")
print(df[df['clasificacion'] == 'no margarita'])


# crear tabla clasificacion

df.to_csv('clasificacion.csv', index=False)

print("Archivo CSV creado exitosamente.")
