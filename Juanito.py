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

# grafica

# Cargar el archivo CSV en un dataframe
df = pd.read_csv('clasificacion.csv')

# Gráfico de barras
sns.countplot(x='clasificacion', data=df)
plt.title('Cantidad de flores clasificadas como "margarita" y "no margarita"')
plt.show()

## se puede concluir con la grafica de barras que hay una mayor cantidad llegando aproximadamente a 140 flores que no son margaritas, 
## y apenas aproximadamente 10 siendo margarta

# Gráfico de dispersión
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='clasificacion', data=df)
plt.title('Relación entre longitud y ancho del sépalo')
plt.show()

## Las flores de "margarita" (representadas en azul en el gráfico de dispersión) generalmente se encuentran en
## la parte superior del gráfico, donde la longitud del sépalo (eje x) es mayor o igual a 5.1 y el ancho del sépalo (eje y) es mayor o igual a 3.5.
## Las flores de "no margarita" (representadas en naranja) tienden a estar en la parte inferior derecha del gráfico, donde la longitud del sépalo 
## es mayor de 5.1 y el ancho del sépalo es menor de 3.5. y su gran cantidad afirmando la conclusion de barras.


# Gráfico de violín
sns.violinplot(x='clasificacion', y='sepal length (cm)', data=df)
plt.title('Distribución de la longitud del sépalo para las categorías "margarita" y "no margarita"')
plt.show()

## Para las flores clasificadas como "margarita", el violín muestra una distribución de longitudes del sépalo que tiende a tener una mediana mayor o 
## igual a 5.1 cm, lo que confirma la restricción. Además, se observa una mayor dispersión en las longitudes del sépalo de las "margaritas", con algunos valores extremadamente altos.
## Para las flores clasificadas como "no margarita", el violín muestra que la mayoría de ellas tienen una mediana de longitud del sépalo menor de 5.1 cm, lo que también concuerda con 
## la restricción. La distribución es más compacta, lo que indica una menor variabilidad en las longitudes del sépalo para las "no margaritas".



