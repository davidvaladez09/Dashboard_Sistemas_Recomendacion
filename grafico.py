import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
from tkinter.scrolledtext import ScrolledText
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

archivo = 'D:\\Documentos\\8vo\\Clasificacion Inteligente de Datos\\Dataset\\rotten_tomatoes_movies.csv'
n_cluster = 5

def k_means():
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

    # Crear una nueva figura para la gráfica
    fig = Figure(figsize=(5, 4), dpi=100)
    plot = fig.add_subplot(111)

    # Visualizar los resultados del clustering en la nueva figura
    plot.scatter(df['tomatometer_rating'], df['audience_rating'], c=df['cluster'], cmap='viridis')
    plot.set_xlabel('Tomatometer Rating')
    plot.set_ylabel('Audience Rating')
    plot.set_title('Clustering de Películas')

    # Dibujar la gráfica en el área de la ventana principal
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    # Imprimir los centros de los clusters
    print("Centros de los Clusters:")
    print(kmeans.cluster_centers_)

    # Calcular el valor de la silueta
    silhouette_avg = silhouette_score(X_normalized, kmeans.labels_)
    print("Evaluación de Silueta:", silhouette_avg)

    # Imprimir los centros de los clusters
    cluster_centers_text = "Centros de los Clusters:\n" + str(kmeans.cluster_centers_)
    silhouette_score_text = "Evaluación de Silueta: " + str(silhouette_score(X_normalized, kmeans.labels_))

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, cluster_centers_text + "\n" + silhouette_score_text)

def agrupar_genero():
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

    # Crear una nueva figura para la gráfica
    fig = Figure(figsize=(5, 4), dpi=100)
    plot = fig.add_subplot(111)

    # Visualizar los resultados del clustering
    plot.scatter(df['tomatometer_rating'], df['audience_rating'], c=df['cluster'], cmap='viridis')
    plot.set_xlabel('Tomatometer Rating')
    plot.set_ylabel('Audience Rating')
    plot.set_title('Clustering de Películas por Género')

    # Dibujar la gráfica en el área de la ventana principal
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    # Imprimir los centros de los clusters
    cluster_centers_text = "Centros de los Clusters:\n" + str(kmeans.cluster_centers_)
    silhouette_score_text = "Evaluación de Silueta: " + str(silhouette_score(Y, kmeans.labels_))

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, cluster_centers_text + "\n" + silhouette_score_text)
    pass

def segmento_audiencia():
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

    # Crear una nueva figura para la gráfica
    fig = Figure(figsize=(5, 4), dpi=100)
    plot = fig.add_subplot(111)

    # Visualizar los resultados del clustering
    plot.scatter(df['audience_rating'], df['audience_count'], c=df['cluster'], cmap='viridis')
    plot.set_xlabel('Audience Rating')
    plot.set_ylabel('Audience Count')
    plot.set_title('Segmentación de Audiencia')

    # Dibujar la gráfica en el área de la ventana principal
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    # Imprimir los centros de los clusters
    print("Centros de los Clusters:")
    print(kmeans.cluster_centers_)

    # Calcular el valor de la silueta
    silhouette_avg = silhouette_score(X_normalized, kmeans.labels_)
    print("Evaluación de Silueta:", silhouette_avg)

    # Imprimir los centros de los clusters
    cluster_centers_text = "Centros de los Clusters:\n" + str(kmeans.cluster_centers_)
    silhouette_score_text = "Evaluación de Silueta: " + str(silhouette_score(X_normalized, kmeans.labels_))

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, cluster_centers_text + "\n" + silhouette_score_text)
    pass

def clasificar_peliculas(genero):
    # Cargar el conjunto de datos desde el archivo CSV
    df = pd.read_csv(archivo)

    X = df[['Comedy', 'Drama', 'Action_&_Adventure', 'Science_Fiction_&_Fantasy', 'Romance', 'Classics', 'Kids_&_Family', 'Mystery_&_Suspense', 'Western', 'Art_House_&_International', 'Faith_&_Spirituality', 'Documentary', 'Special_Interest', 'audience_rating']]  # Aquí agregamos características relevantes como el género

    # Normalizar los datos para mejorar el rendimiento del algoritmo K-Means
    X_normalized = (X - X.mean()) / X.std()


    # Entrenar el modelo K-Means
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    kmeans.fit(X_normalized)


    # Agregar las etiquetas de cluster al DataFrame original
    df['cluster'] = kmeans.labels_

    # Filtrar el DataFrame según el género seleccionado
    filtered_df = df[df[genero] == 1]

    # Crear una nueva figura para la gráfica
    fig = Figure(figsize=(5, 4), dpi=100)
    plot = fig.add_subplot(111)

    # Visualizar los resultados del clustering
    plot.scatter(df[genero], df['audience_rating'], c=df['cluster'], cmap='viridis') # Visualizamos el género Action_&_Adventure y el tiempo de ejecución
    plot.set_xlabel('Tomatometer Rating')
    plot.set_ylabel('Audience Rating')
    plot.set_title(f'Clustering de Películas por Género: {genero}')

    # Dibujar la gráfica en el área de la ventana principal
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    # Imprimir los centros de los clusters
    print("Centros de los Clusters:")
    print(kmeans.cluster_centers_)

    # Calcular el valor de la silueta
    silhouette_avg = silhouette_score(X_normalized, kmeans.labels_)
    print("Evaluación de Silueta:", silhouette_avg)

    # Imprimir los centros de los clusters
    cluster_centers_text = "Centros de los Clusters:\n" + str(kmeans.cluster_centers_)
    silhouette_score_text = "Evaluación de Silueta: " + str(silhouette_score(X_normalized, kmeans.labels_))

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, cluster_centers_text + "\n" + silhouette_score_text)
    pass

def ejecutar_funcion_4():
     # Diccionario de géneros
    dic_genero = {1:'Comedy', 2:'Drama', 3:'Action_&_Adventure', 4:'Science_Fiction_&_Fantasy', 5:'Romance', 6:'Classics', 7:'Kids_&_Family', 8:'Mystery_&_Suspense', 9:'Western', 10:'Art_House_&_International', 11:'Faith_&_Spirituality', 12:'Documentary', 13:'Special_Interest'}
    
    # Mostrar el diccionario de géneros dentro del diálogo simple
    genero = simpledialog.askinteger("Ingresar Género", f"Ingresa el número de género: {dic_genero}", parent=root)

    # Verificar si se ingresó un género válido
    if genero is not None:
        nuevo_genero = dic_genero.get(genero)

        # Verificar si el número de género ingresado es válido
        if nuevo_genero is not None:
            clasificar_peliculas(nuevo_genero)
        else:
            print("Número de género inválido.")

def analisis_contenido():
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

    # Imprimir los centros de los clusters
    cluster_centers_text = "Centros de los Clusters:\n" + str(kmeans.cluster_centers_)
    silhouette_score_text = "Evaluación de Silueta: " + str(silhouette_score(X, kmeans.labels_))

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, cluster_centers_text + "\n" + silhouette_score_text)

    pass

# Función para limpiar el área de la gráfica
def clear_plot():
    for widget in plot_frame.winfo_children():
        widget.destroy()

# Configuración de la ventana principal
root = tk.Tk()
root.title("Ventana con gráficas")

# Crear el marco para los botones
button_frame = ttk.Frame(root)
button_frame.pack(side="left", fill="y")

# Crear los botones
button1 = ttk.Button(button_frame, text="K Means", command=k_means)
button1.pack(fill="x", padx=5, pady=5)

button2 = ttk.Button(button_frame, text="Agrupar Genero", command=agrupar_genero)
button2.pack(fill="x", padx=5, pady=5)

button3 = ttk.Button(button_frame, text="Segmentar Audiencia", command=segmento_audiencia)
button3.pack(fill="x", padx=5, pady=5)

button4 = ttk.Button(button_frame, text="Clasificar Peliculas por Genero", command=ejecutar_funcion_4)
button4.pack(fill="x", padx=5, pady=5)

button5 = ttk.Button(button_frame, text="Analisis Contenido", command=analisis_contenido)
button5.pack(fill="x", padx=5, pady=5)

# Crear el marco para la gráfica
plot_frame = ttk.Frame(root)
plot_frame.pack(side="right", fill="both", expand=True)

# Crear el widget de Texto para mostrar los resultados debajo de la gráfica
result_text = ScrolledText(root, wrap=tk.WORD, width=40, height=10)
result_text.pack(side="bottom", padx=5, pady=5)

# Botón para limpiar la gráfica
clear_button = ttk.Button(root, text="Limpiar Gráfica", command=clear_plot)
clear_button.pack(side="bottom", pady=5)

# Lanzar el bucle principal de la ventana
root.mainloop()
