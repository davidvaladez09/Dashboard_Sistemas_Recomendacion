import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Función para calcular la distancia entre un punto y los centroides
def calcular_distancia(punto, centroides):
    return np.linalg.norm(punto - centroides, axis=1)  # Resta el punto individual de cada fila de centroides. Retorna diferencia entre el punto y un centroide

# axis es para especificar el eje a lo largo del cual se realiza una operación

# Función para asignar cada punto al cluster más cercano
def asignar_cluster(caracteristicas, centroides):
    distancias = np.apply_along_axis(calcular_distancia, 1, caracteristicas, centroides) # calcula la distancia entre un punto y un conjunto de centroides
    return np.argmin(distancias, axis=1) # Retorna arreglo que contiene el índice del centroide más cercano a cada punto

# Función para actualizar los centroides
def actualizar_centroides(caracteristicas, clusters_asignados, k):
    nuevos_centroides = [] # almacenarán los nuevos centroides
    for i in range(k): #  Itera sobre cada índice de cluster
        cluster_i = caracteristicas[clusters_asignados == i] # Selecciona los puntos del conjunto de características que están asignados al cluster
        nuevos_centroides.append(cluster_i.mean(axis=0)) # Calcula la media de los puntos en el cluster i, lo que resulta en un nuevo centroide para el cluster 
    return np.array(nuevos_centroides) # Retorna el resultado de la función como arreglo

# Función principal para el algoritmo k-means
def kmeans(archivo):
    # Leer los datos desde un archivo usando pandas
    datos = pd.read_csv(archivo)

    # Tomar solo el 70% de los datos para el análisis
    datos_muestra = datos.sample(frac=0.7, random_state=42)

    # Seleccionar las características relevantes para el clustering (Columnas relevantes)
    caracteristicas = datos_muestra[['audience_rating', 'tomatometer_rating', 'runtime_in_minutes']]

    # Asignar el numero de clusters
    k = 4

    # Paso 1: INICIALIZAR los centroides de forma aleatoria
    centroides = caracteristicas.sample(n=k).values

    # Paso 2: ASIGNAR PUNTOS cada punto al CLUSTER más cercano
    clusters_asignados = asignar_cluster(caracteristicas.values, centroides)

    # Paso 3: ACTUALIZAR LOS CENTROIDES
    centroides_actualizados = actualizar_centroides(caracteristicas.values, clusters_asignados, k)

    # Paso 4: REPETIR LOS PASOS 2-3 hasta que los centroides no cambien significativamente
    while not np.allclose(centroides, centroides_actualizados):
        centroides = centroides_actualizados
        clusters_asignados = asignar_cluster(caracteristicas.values, centroides)
        centroides_actualizados = actualizar_centroides(caracteristicas.values, clusters_asignados, k)

    # Paso 5: MOSTRAR LOS RESULTADOS de los centroides y los clusters
    print("Centroides finales:")
    print(centroides_actualizados)
    print("\nClusters asignados:")
    print(clusters_asignados)

    # Mostrar la información de cada cluster
    for cluster_id in range(k):
        print(f"\nCluster {cluster_id + 1}:")
        cluster_data = datos_muestra[clusters_asignados == cluster_id]
        print(cluster_data)
    
    # Paso 5: PROBAR EL ALGORITMO
    # Calcular el coeficiente de silueta
    coeficiente_silueta = silhouette_score(caracteristicas, clusters_asignados)
    print(f"\nCoeficiente de Silueta: {coeficiente_silueta}")

    # Paso 6: Graficar los resultados
    plt.figure(figsize=(8, 6))
    for cluster_id in range(k):
        cluster_data = datos_muestra[clusters_asignados == cluster_id]
        plt.scatter(cluster_data['runtime_in_minutes'], cluster_data['tomatometer_rating'], label=f'Cluster {cluster_id + 1}')
    plt.scatter(centroides[:, 0], centroides[:, 1], marker='*', color='black', s=150, label='Centroides')
    plt.xlabel('Duración en minutos')
    plt.ylabel('Tomatometer Rating')
    plt.title('Clusters de Películas')
    plt.legend()
    plt.grid(True)
    plt.show()

archivo = 'D:\\Documentos\\8vo\\Clasificacion Inteligente de Datos\\Dataset\\RottenTomatoesMovies1.csv'

# Llamar a la función 
kmeans(archivo)
