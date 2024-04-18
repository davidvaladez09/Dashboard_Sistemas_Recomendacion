import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

# Leer el archivo CSV con pandas
df = pd.read_csv('D:\\Documentos\\8vo\\Clasificacion Inteligente de Datos\\Dataset\\rotten_tomatoes_movies.csv')


# Generar datos de ejemplo para las otras gráficas
x = np.linspace(0, 10, 100)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(x)
y5 = np.log(x + 1)
y6 = np.sqrt(x)
y7 = x ** 2
y8 = np.random.normal(size=100)
y9 = np.random.uniform(size=100)

# Crear el dashboard con las gráficas
plt.figure(figsize=(15, 15))

# Gráfica 1 (Generada a partir del archivo CSV)
plt.subplot(3, 3, 1)
df.groupby('content_rating')['audience_count'].sum().plot(kind='bar')
plt.title('Suma de Audiencia por Clasificación de Contenido')
plt.xlabel('Clasificación de Contenido')
plt.ylabel('Suma de Audiencia')

# Gráfica 2
plt.subplot(3, 3, 2)
genre_counts = df['tomatometer_status'].value_counts()
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%')
plt.title('Porcentaje de Audiencia por Tomatometer Status')

# Gráfica 3
plt.subplot(3, 3, 3)
#plt.plot(x, y3)
#plt.title('Gráfica 3: Tan(x)')
genre_counts = df['content_rating'].value_counts()
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%')
plt.title('Porcentaje de Audiencia por Clasificacion')

# Gráfica 4
plt.subplot(3, 3, 4)
plt.scatter(df['runtime'], df['audience_rating'])
plt.title('Dispersión de Audiencia vs Duración en Minutos')
plt.xlabel('Duración en Minutos')
plt.ylabel('Audiencia')

# Gráfica 5
plt.subplot(3, 3, 5)
sns.regplot(data=df, x='runtime', y='audience_rating')
plt.title('Regresión Lineal de Audiencia vs Duración en Minutos')
plt.xlabel('Duración en Minutos')
plt.ylabel('Audiencia')

# Gráfica 6 (Gráfico de contorno) Representa los límites de los clusters en un espacio bidimensional, mostrando regiones de densidad alta y baja.
plt.subplot(3, 3, 6)
sns.kdeplot(data=df, x='tomatometer_rating', y='audience_rating', cmap='viridis', fill=True)
plt.title('Gráfico de Contorno de Audiencia vs Calificación de Críticos')
plt.xlabel('Calificación de Críticos')
plt.ylabel('Audiencia')

# Gráfica 7 (Histograma)
plt.subplot(3, 3, 7)
sns.histplot(data=df, x='audience_rating', bins=20, kde=True)
plt.title('Gráfica 7: Histograma de Audiencia')
plt.xlabel('Audiencia')
plt.ylabel('Frecuencia')

# Gráfica 8 (Diagrama de Voronoi) Divide el espacio en regiones que están más cerca de cada centroide de cluster que de cualquier otro centroide.
# Gráfica 8 (Diagrama de Voronoi)
plt.subplot(3, 3, (8, 9))  # Subplot que ocupa dos columnas
points = np.random.rand(10, 2)  # Generar puntos aleatorios
vor = Voronoi(points)  # Calcular diagrama de Voronoi
voronoi_plot_2d(vor, show_vertices=False)  # Mostrar el diagrama de Voronoi
plt.title('Gráfica 8: Diagrama de Voronoi')

# Gráfica 9
'''plt.subplot(3, 3, 9)
sns.histplot(y9, kde=True)
plt.title('Gráfica 9: Distribución Uniforme')'''

plt.tight_layout()
plt.show()