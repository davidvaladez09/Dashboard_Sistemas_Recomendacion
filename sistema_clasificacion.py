import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MultiLabelBinarizer

def k_means(archivo, n_cluster):
    # Cargar el conjunto de datos desde el archivo CSV
    df = pd.read_csv(archivo)

    # Seleccionar las características relevantes para clustering
    X = df[['tomatometer_rating', 'audience_rating', 'runtime']]

    # Normalizar los datos para mejorar el rendimiento del algoritmo K-Means
    X_normalized = (X - X.mean()) / X.std()

    # Entrenar el modelo K-Means
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    kmeans.fit(X_normalized)

    # Agregar las etiquetas de cluster al DataFrame original
    df['cluster'] = kmeans.labels_

    # Visualizar los resultados del clustering
    plt.scatter(df['tomatometer_rating'], df['audience_rating'], c=df['cluster'], cmap='viridis')
    plt.xlabel('Tomatometer Rating')
    plt.ylabel('Audience Rating')
    plt.title('Clustering de Películas')
    plt.show()

    # Imprimir los centros de los clusters
    print("Centros de los Clusters:")
    print(kmeans.cluster_centers_)

    # Calcular el valor de la silueta
    silhouette_avg = silhouette_score(X_normalized, kmeans.labels_)
    print("Evaluación de Silueta:", silhouette_avg)

#----------------------------------------------------------------------------------------------#----------------------------------------------------------------------------------------------

def agrupar_genero(archivo, n_cluster):

    # Cargar el conjunto de datos desde el archivo CSV
    df = pd.read_csv(archivo)

    # Convertir las características de género en binarias
    genre_columns = ['Comedy', 'Drama', 'Action_&_Adventure', 'Science_Fiction_&_Fantasy', 'Romance', 'Classics', 'Kids_&_Family', 'Mystery_&_Suspense', 'Western', 'Art_House_&_International', 'Faith_&_Spirituality', 'Documentary', 'Special_Interest']
    for genre in genre_columns:
        df[genre] = df['genres'].str.contains(genre).astype(int)

    # Seleccionar las características relevantes para clustering
    Y = df[genre_columns]

    # Entrenar el modelo K-Means
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    kmeans.fit(Y)

    # Agregar las etiquetas de cluster al DataFrame original
    df['cluster'] = kmeans.labels_

    # Visualizar los resultados del clustering
    plt.scatter(df['tomatometer_rating'], df['audience_rating'], c=df['cluster'], cmap='viridis')
    plt.xlabel('Tomatometer Rating')
    plt.ylabel('Audience Rating')
    plt.title('Clustering de Películas por Género')
    plt.show()

    # Imprimir los centros de los clusters
    print("Centros de los Clusters:")
    print(kmeans.cluster_centers_)

    # Calcular el valor de la silueta
    silhouette_avg = silhouette_score(Y, kmeans.labels_)
    print("Evaluación de Silueta:", silhouette_avg)

def agrupar_estudio(archivo, n_cluster):
    # Cargar el conjunto de datos desde el archivo CSV
    df = pd.read_csv(archivo)

    # Preprocesamiento de datos
    # Convertir la columna 'production_company' en características numéricas utilizando codificación one-hot
    mlb = MultiLabelBinarizer()
    production_companies = df['production_company'].apply(lambda x: [x.strip() for x in x.split('/')])
    company_encoded = pd.DataFrame(mlb.fit_transform(production_companies), columns=mlb.classes_, index=df.index)

    # Combinar las características codificadas con el DataFrame original
    df_encoded = pd.concat([df, company_encoded], axis=1)

    # Seleccionar las características relevantes para clustering
    X = df_encoded.drop(['production_company', 'tomatometer_rating', 'audience_rating', 'runtime'], axis=1)

    # Normalizar los datos para mejorar el rendimiento del algoritmo K-Means
    X_normalized = (X - X.mean()) / X.std()

    # Entrenar el modelo K-Means
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    kmeans.fit(X_normalized)

    # Agregar las etiquetas de cluster al DataFrame original
    df['cluster'] = kmeans.labels_

    # Calcular el valor de la silueta
    silhouette_avg = silhouette_score(X_normalized, kmeans.labels_)
    print("Evaluación de Silueta:", silhouette_avg)

    # Visualizar los resultados del clustering
    plt.figure(figsize=(10, 6))
    cluster_counts = df.groupby('cluster')['production_company'].value_counts().unstack(fill_value=0)
    cluster_counts.plot(kind='bar', stacked=True)
    plt.xlabel('Cluster')
    plt.ylabel('Número de películas')
    plt.title('Distribución de películas por estudio en cada cluster')
    plt.legend(title='Estudio', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return df, silhouette_avg

def segmento_audiencia(archivo, n_cluster):
    # Cargar el conjunto de datos desde el archivo CSV
    df = pd.read_csv(archivo)

    # Seleccionar las características relevantes para clustering
    X = df[['audience_rating', 'audience_count']]  # Usamos la audiencia_rating y audience_count para segmentar la audiencia

    # Normalizar los datos para mejorar el rendimiento del algoritmo K-Means
    X_normalized = (X - X.mean()) / X.std()

    # Entrenar el modelo K-Means
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    kmeans.fit(X_normalized)

    # Agregar las etiquetas de cluster al DataFrame original
    df['cluster'] = kmeans.labels_

    # Visualizar los resultados del clustering
    plt.scatter(df['audience_rating'], df['audience_count'], c=df['cluster'], cmap='viridis')
    plt.xlabel('Audience Rating')
    plt.ylabel('Audience Count')
    plt.title('Segmentación de Audiencia')
    plt.show()

    # Imprimir los centros de los clusters
    print("Centros de los Clusters:")
    print(kmeans.cluster_centers_)

    # Calcular el valor de la silueta
    silhouette_avg = silhouette_score(X_normalized, kmeans.labels_)
    print("Evaluación de Silueta:", silhouette_avg)

def clasificar_peliculas(archivo, n_cluster, genero):
    # Cargar el conjunto de datos desde el archivo CSV
    df = pd.read_csv(archivo)

    # Seleccionar las características relevantes para clustering
    X = df[['Comedy', 'Drama', 'Action_&_Adventure', 'Science_Fiction_&_Fantasy', 'Romance', 'Classics', 'Kids_&_Family', 'Mystery_&_Suspense', 'Western', 'Art_House_&_International', 'Faith_&_Spirituality', 'Documentary', 'Special_Interest', 'audience_rating']]  # Aquí agregamos características relevantes como el género

    # Normalizar los datos para mejorar el rendimiento del algoritmo K-Means
    X_normalized = (X - X.mean()) / X.std()

    # Entrenar el modelo K-Means
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    kmeans.fit(X_normalized)

    # Agregar las etiquetas de cluster al DataFrame original
    df['cluster'] = kmeans.labels_

    # Visualizar los resultados del clustering
    plt.scatter(df[genero], df['audience_rating'], c=df['cluster'], cmap='viridis')  # Visualizamos el género Action_&_Adventure y el tiempo de ejecución
    plt.xlabel(genero)
    plt.ylabel('Runtime (minutes)')
    plt.title('Clasificación de Películas')
    plt.show()

    # Imprimir los centros de los clusters
    print("Centros de los Clusters:")
    print(kmeans.cluster_centers_)

    # Calcular el valor de la silueta
    silhouette_avg = silhouette_score(X_normalized, kmeans.labels_)
    print("Evaluación de Silueta:", silhouette_avg)

def analisis_contenido(archivo, n_cluster):
    df = pd.read_csv(archivo)

    # Preprocesamiento de datos
    # Seleccionar el campo de sinopsis o críticas como características relevantes para clustering
    corpus = df['critics_consensus']  # Por ejemplo, aquí usaremos el campo 'critics_consensus'
    corpus = corpus.fillna('')  # Manejo de valores nulos

    # Convertir el corpus de texto en una matriz TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)

    # Entrenar el modelo K-Means
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    kmeans.fit(X)

    # Agregar las etiquetas de cluster al DataFrame original
    df['cluster'] = kmeans.labels_

    # Visualizar los resultados del clustering
    # (En este ejemplo, no podemos visualizar directamente el clustering debido a la naturaleza de los datos de texto)
    # Sin embargo, puedes explorar los grupos resultantes y analizar los temas recurrentes o patrones en el contenido de las películas

    # Imprimir los centros de los clusters
    print("Centros de los Clusters:")
    print(kmeans.cluster_centers_)

    # Calcular el valor de la silueta
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    print("Evaluación de Silueta:", silhouette_avg)

def listar_datos_unicos(archivo_csv, columna):
    # Leer el archivo CSV
    datos = pd.read_csv(archivo_csv)
    
    # Extraer la columna especificada
    columna_datos = datos[columna]
    
    # Obtener valores únicos de la columna
    valores_unicos = columna_datos.unique()
    
    # Imprimir los datos únicos
    print("Datos únicos en la columna '{}':".format(columna))
    for valor in valores_unicos:
        print(valor)


archivo = 'D:\\Documentos\\8vo\\Clasificacion Inteligente de Datos\\Dataset\\rotten_tomatoes_movies.csv'
n_cluster = 5

columna = "production_company"   # Nombre de la columna que deseas listar


while True:

    print('\nMenu\n1. K-Means.\n2. Agrupar Genero\n3. Segmentar Audiencia. \n4. Clasificar Peliculas por Genero. \n5. Analisis Contenido.')

    opcion = int(input('Ingresa una opcion: '))

    if opcion == 1:
        k_means(archivo, n_cluster)

    elif opcion == 2:
        agrupar_genero(archivo, n_cluster)

    elif opcion == 3:
        segmento_audiencia(archivo, n_cluster)

    elif opcion == 4:
        dic_genero = {1:'Comedy', 2:'Drama', 3:'Action_&_Adventure', 4:'Science_Fiction_&_Fantasy', 5:'Romance', 6:'Classics', 7:'Kids_&_Family', 8:'Mystery_&_Suspense', 9:'Western', 10:'Art_House_&_International', 11:'Faith_&_Spirituality', 12:'Documentary', 13:'Special_Interest'}
        nuevo_genero = ''
        
        print('\nClasificar peliculas por genero\n')
        
        for indice, valor in dic_genero.items():
            print(indice," ", valor)
        
        genero = int(input('Ingresa un genero: '))

        for indice, valor in dic_genero.items():
            if genero == indice:
                nuevo_genero = valor

        clasificar_peliculas(archivo, n_cluster, nuevo_genero)

    elif opcion == 5:
        analisis_contenido(archivo, n_cluster)

    elif opcion != 6:
        print('\nSALIENDO')
        break