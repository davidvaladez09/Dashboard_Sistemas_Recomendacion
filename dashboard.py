import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.widgets import Button

# Leer el archivo CSV con pandas (Este código no se ha modificado)
df = pd.read_csv('D:\\Documentos\\8vo\\Clasificacion Inteligente de Datos\\Dataset\\rotten_tomatoes_movies.csv')

# Generar datos de ejemplo para las otras gráficas (Este código no se ha modificado)
x = np.linspace(0, 10, 100)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(x)
y5 = np.log(x + 1)
y6 = np.sqrt(x)
y7 = x ** 2
y8 = np.random.normal(size=100)
y9 = np.random.uniform(size=100)

# Función para mostrar la gráfica de Voronoi
def mostrar_voronoi(event):
    plt.subplot(3, 3, 8)
    plt.cla()  # Limpiar la subtrama
    points = np.random.rand(10, 2)  # Generar puntos aleatorios
    vor = Voronoi(points)  # Calcular diagrama de Voronoi
    voronoi_plot_2d(vor, show_vertices=False)  # Mostrar el diagrama de Voronoi
    plt.title('Gráfica 8: Diagrama de Voronoi')
    plt.subplots_adjust(left=0.1, bottom=0.3)  # Ajustar el espacio para el botón
    plt.show()

# Crear el dashboard con las gráficas (Este código se ha modificado)
plt.figure(figsize=(15, 15))

# Gráfica 1 a 7 (Este código no se ha modificado)
for i in range(1, 8):
    plt.subplot(3, 3, i)
    if i == 1:
        df.groupby('content_rating')['audience_count'].sum().plot(kind='bar')
        plt.title('Suma de Audiencia por Clasificación de Contenido')
        plt.xlabel('Clasificación de Contenido')
        plt.ylabel('Suma de Audiencia')
    elif i == 2:
        genre_counts = df['tomatometer_status'].value_counts()
        plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%')
        plt.title('Porcentaje de Audiencia por Tomatometer Status')
    elif i == 3:
        genre_counts = df['content_rating'].value_counts()
        plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%')
        plt.title('Porcentaje de Audiencia por Clasificacion')
    elif i == 4:
        plt.scatter(df['runtime'], df['audience_rating'])
        plt.title('Dispersión de Audiencia vs Duración en Minutos')
        plt.xlabel('Duración en Minutos')
        plt.ylabel('Audiencia')
    elif i == 5:
        sns.regplot(data=df, x='runtime', y='audience_rating')
        plt.title('Regresión Lineal de Audiencia vs Duración en Minutos')
        plt.xlabel('Duración en Minutos')
        plt.ylabel('Audiencia')
    elif i == 6:
        sns.kdeplot(data=df, x='tomatometer_rating', y='audience_rating', cmap='viridis', fill=True)
        plt.title('Gráfico de Contorno de Audiencia vs Calificación de Críticos')
        plt.xlabel('Calificación de Críticos')
        plt.ylabel('Audiencia')
    elif i == 7:
        sns.histplot(data=df, x='audience_rating', bins=20, kde=True)
        plt.title('Gráfica 7: Histograma de Audiencia')
        plt.xlabel('Audiencia')
        plt.ylabel('Frecuencia')

# Agregar botón para mostrar la gráfica de Voronoi
ax_button = plt.axes([0.3, 0.3, 0.3, 0.10])  # Definir la posición y tamaño del botón
button = Button(ax_button, 'Mostrar Voronoi')
button.on_clicked(mostrar_voronoi)  # Asignar la función al botón

plt.tight_layout()
plt.show()



# Gráfica 9
'''plt.subplot(3, 3, 9)
sns.histplot(y9, kde=True)
plt.title('Gráfica 9: Distribución Uniforme')'''
