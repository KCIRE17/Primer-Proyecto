import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Cargar el DataFrame limpio
try:
    df_limpio = pd.read_csv('IMDB_Dataset_limpio.csv')
    print("Datos limpios cargados exitosamente.")
except FileNotFoundError:
    print("Error: El archivo 'IMDB_Dataset_limpio.csv' no se encuentra.")
    print("Asegúrate de haber ejecutado el script de limpieza de datos primero.")
    exit()

# 2. Definir los datos (X) y las etiquetas (y)
X = df_limpio['review']
y = df_limpio['sentiment']

# 3. Vectorizar el texto usando TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = tfidf_vectorizer.fit_transform(X)

print("\nTexto vectorizado con TF-IDF.")
print(f"Forma de la matriz de características: {X_vectorized.shape}")

# 4. Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

print(f"\nConjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras")

# 5. Entrenar el modelo Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

print("\nModelo Naive Bayes ha sido entrenado")

# 6. Evaluar el rendimiento del modelo
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy:.2f}")

print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred))