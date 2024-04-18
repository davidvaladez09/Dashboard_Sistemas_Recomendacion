import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
import numpy as np



def plotSpotify():
    archivo = 'D:\Documentos\8vo\Clasificacion Inteligente de Datos\Dataset\spotify-2023.csv'

    # Cargar datos desde el archivo CSV
    datos = pd.read_csv(archivo, encoding='latin1')

    # Listar datos
    print("Datos Originales:")
    print(datos.head())

    # Barajar datos
    datos_barajados = datos.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separar datos en entrenamiento (70%) y pruebas (30%)
    datos_entrenamiento, datos_prueba = train_test_split(datos_barajados, test_size=0.3, random_state=42)

    # Mostrar los datos de entrenamiento y prueba
    print("\nDatos de Entrenamiento:")
    print(datos_entrenamiento)
    print("\nDatos de Prueba:")
    print(datos_prueba)

    # Realizar regresión lineal
    slope, intercept, r_value, p_value, std_err = linregress(datos_entrenamiento['artist_count'], datos_entrenamiento['released_year'])

    # Graficar utilizando seaborn
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=datos_entrenamiento, x='artist_count', y='released_year')
    plt.title('Plot Spotify')
    plt.xlabel('Variable X, artist_count')
    plt.ylabel('Variable Y, released_year')

    # Añadir la línea de regresión
    plt.plot(datos_entrenamiento['artist_count'], slope * datos_entrenamiento['artist_count'] + intercept, color='red')

    # Añadir texto con el valor de r
    plt.text(0.1, 0.9, f'r = {r_value:.2f}', ha='center', va='center', transform=plt.gca().transAxes)

    plt.grid(True)
    plt.show()


def plotAirlines():
    archivo = 'D:\Documentos\8vo\Clasificacion Inteligente de Datos\Dataset\Airlines.csv'

    # Cargar datos desde el archivo CSV
    datos = pd.read_csv(archivo, encoding='latin1')

    # Listar datos
    print("Datos Originales:")
    print(datos.head())

    # Barajar datos
    datos_barajados = datos.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separar datos en entrenamiento (70%) y pruebas (30%)
    datos_entrenamiento, datos_prueba = train_test_split(datos_barajados, test_size=0.3, random_state=42)

    # Mostrar los datos de entrenamiento y prueba
    print("\nDatos de Entrenamiento:")
    print(datos_entrenamiento)
    print("\nDatos de Prueba:")
    print(datos_prueba)

    # Realizar regresión lineal
    slope, intercept, r_value, p_value, std_err = linregress(datos_entrenamiento['Length'], datos_entrenamiento['Time'])

    # Graficar utilizando seaborn
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=datos_entrenamiento, x='Length', y='Time')
    plt.title('Plot Airlines')
    plt.xlabel('Variable X, Length')
    plt.ylabel('Variable Y, Time')

    # Añadir la línea de regresión
    plt.plot(datos_entrenamiento['Length'], slope * datos_entrenamiento['Length'] + intercept, color='red')

    # Añadir texto con el valor de r
    plt.text(0.1, 0.9, f'r = {r_value:.2f}', ha='center', va='center', transform=plt.gca().transAxes)

    plt.grid(True)
    plt.show()


def plotSalarios():
    archivo = 'D:\Documentos\8vo\Clasificacion Inteligente de Datos\Dataset\Salary_Data.csv'

    # Cargar datos desde el archivo CSV
    datos = pd.read_csv(archivo, encoding='latin1')

    # Listar datos
    print("Datos Originales:")
    print(datos.head())

    # Barajar datos
    datos_barajados = datos.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separar datos en entrenamiento (70%) y pruebas (30%)
    datos_entrenamiento, datos_prueba = train_test_split(datos_barajados, test_size=0.3, random_state=42)

    # Mostrar los datos de entrenamiento y prueba
    print("\nDatos de Entrenamiento:")
    print(datos_entrenamiento)
    print("\nDatos de Prueba:")
    print(datos_prueba)

    # Realizar regresión lineal
    slope, intercept, r_value, p_value, std_err = linregress(datos_entrenamiento['Years_of_Experience'], datos_entrenamiento['Salary'])

    # Graficar utilizando seaborn
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=datos_entrenamiento, x='Years_of_Experience', y='Salary')
    plt.title('Plot Salarios')
    plt.xlabel('Variable X, Years_of_Experience')
    plt.ylabel('Variable Y, Salary')

    # Añadir la línea de regresión
    plt.plot(datos_entrenamiento['Years_of_Experience'], slope * datos_entrenamiento['Years_of_Experience'] + intercept, color='red')

    # Añadir texto con el valor de r
    plt.text(0.1, 0.9, f'r = {r_value:.2f}', ha='center', va='center', transform=plt.gca().transAxes)

    plt.grid(True)
    plt.show()



def plotDiabetes():
    archivo = 'D:\Documentos\8vo\Clasificacion Inteligente de Datos\Dataset\diabetes_prediction_dataset.csv'
    
    # Cargar datos desde el archivo CSV
    datos = pd.read_csv(archivo, encoding='latin1')

    # Listar datos
    print("Datos Originales:")
    print(datos.head())

    # Barajar datos
    datos_barajados = datos.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separar datos en entrenamiento (70%) y pruebas (30%)
    datos_entrenamiento, datos_prueba = train_test_split(datos_barajados, test_size=0.3, random_state=42)

    # Mostrar los datos de entrenamiento y prueba
    print("\nDatos de Entrenamiento:")
    print(datos_entrenamiento)
    print("\nDatos de Prueba:")
    print(datos_prueba)

    # Realizar regresión lineal
    slope, intercept, r_value, p_value, std_err = linregress(datos_entrenamiento['age'], datos_entrenamiento['blood_glucose_level'])

    # Graficar utilizando seaborn
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=datos_entrenamiento, x='age', y='blood_glucose_level')
    plt.title('Plot Diabetes')
    plt.xlabel('Variable X, age')
    plt.ylabel('Variable Y, blood_glucose_level')

    # Añadir la línea de regresión
    plt.plot(datos_entrenamiento['age'], slope * datos_entrenamiento['age'] + intercept, color='red')
 
    # Añadir texto con el valor de r
    plt.text(0.1, 0.9, f'r = {r_value:.2f}', ha='center', va='center', transform=plt.gca().transAxes)

    plt.grid(True)
    plt.show()

plotAirlines()
#plotSalarios()
#plotSpotify()
#plotDiabetes()