import pandas as pd
import re


try:
    df = pd.read_csv('.\IMDB Dataset.csv')
    print("Datos cargados exitosamente.")
except FileNotFoundError:
    print("Error: El archivo 'IMDB_Dataset.csv' no se encuentra.")
    print("Asegúrate de que el archivo esté en el mismo directorio que tu script.")
    exit()

# Paso 2: Inspeccionar el DataFrame
print("\nPrimeras 5 filas del DataFrame:")
print(df.head())

print("\nInformación del DataFrame:")
print(df.info())

# Paso 3: Definir la función de limpieza de texto
def clean_text(text):
    # Eliminar etiquetas HTML
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)

    # Eliminar puntuación y caracteres especiales
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convertir a minúsculas
    text = text.lower()

    # Eliminar espacios extra
    text = re.sub(' +', ' ', text)

    return text

# Paso 4: Aplicar la limpieza a la columna 'review'
df['review'] = df['review'].apply(clean_text)

# Paso 5: Mostrar la primera reseña limpia para verificar
print("\nPrimera reseña después de la limpieza:")
print(df['review'].iloc[0])

# Paso 6: Guardar el nuevo DataFrame limpio
df.to_csv('IMDB_Dataset_limpio.csv', index=False)
print("\nDataFrame limpio guardado como 'IMDB_Dataset_limpio.csv'")