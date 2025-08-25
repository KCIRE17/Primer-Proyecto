import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Cargar el DataFrame limpio que guardaste
df_limpio = pd.read_csv('IMDB_Dataset_limpio.csv')

# Separar el texto de las reseñas por sentimiento
positive_text = ' '.join(df_limpio[df_limpio['sentiment'] == 'positive']['review'])
negative_text = ' '.join(df_limpio[df_limpio['sentiment'] == 'negative']['review'])

# Generar la nube de palabras para reseñas positivas
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

# Generar la nube de palabras para reseñas negativas
wordcloud_negative = WordCloud(width=800, height=400, background_color='black', colormap='Wistia').generate(negative_text)

# Crear los gráficos para mostrar las nubes de palabras
plt.figure(figsize=(15, 7))

# Gráfico para las reseñas positivas
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Palabras Comunes en Reseñas POSITIVAS', fontsize=15)
plt.axis('off')

# Gráfico para las reseñas negativas
plt.subplot(1, 2, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Palabras Comunes en Reseñas NEGATIVAS', fontsize=15, color='white')
plt.axis('off')
plt.tight_layout()
plt.show()

print("Las nubes de palabras se han generado exitosamente.")