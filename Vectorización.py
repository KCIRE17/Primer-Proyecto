import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Cargar el DataFrame limpio
df_limpio = pd.read_csv('IMDB_Dataset_limpio.csv')

# 2. Definir la columna de texto (reseñas) y la columna de etiquetas (sentimiento)
X = df_limpio['review']
y = df_limpio['sentiment']

# 3. Inicializar el vectorizador TF-IDF
# max_features limita la cantidad de palabras únicas que se considerarán, lo que ayuda a reducir el tamaño del modelo.
# stop_words='english' elimina palabras comunes en inglés que no aportan significado (aunque ya lo hicimos, es buena práctica mantenerlo aquí).
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# 4. Transformar el texto en una matriz de características numéricas
# .fit_transform() aprende el vocabulario y transforma las reseñas en vectores
X_vectorized = tfidf_vectorizer.fit_transform(X)

# 5. Mostrar la forma de la matriz resultante
# La forma es (número de reseñas, número de características)
print("Forma de la matriz vectorizada (reseñas, características):")
print(X_vectorized.shape)