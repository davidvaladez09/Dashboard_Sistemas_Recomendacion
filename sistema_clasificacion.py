import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

def kmeans(archivo):
    data = pd.read_csv(archivo)

    # Seleccionar características relevantes
    features = data[['runtime_in_minutes', 'tomatometer_rating', 'audience_rating']]

    # Normalizar los datos
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Aplicar K-Means
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(scaled_features)

    # Agregar las etiquetas de cluster al DataFrame original
    data['cluster'] = kmeans.labels_

    # Mostrar los centroides
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids, columns=features.columns)
    print("Centroides:")
    print(centroids_df)

    # Mostrar los grupos de datos
    print("Grupos de datos:")
    for cluster_id in data['cluster'].unique():
        print(f"Grupo {cluster_id}:")
        print(data[data['cluster'] == cluster_id])

    # Calcular el coeficiente de silueta
    coeficiente_silueta = silhouette_score(scaled_features, kmeans.labels_)
    print(f"Coeficiente de Silueta: {coeficiente_silueta}")

    # Visualizar los clusters
    plt.scatter(data['runtime_in_minutes'], data['audience_rating'], c=data['cluster'], cmap='viridis')
    plt.xlabel('Runtime (minutes)')
    plt.ylabel('Audience Rating')
    plt.title('K-Means Clustering')
    plt.show()


archivo = 'D:\\Documentos\\8vo\\Clasificacion Inteligente de Datos\\Dataset\\RottenTomatoesMovies1.csv'

kmeans(archivo)



