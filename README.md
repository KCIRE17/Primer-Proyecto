# Análisis de sentimientos de reseñas de películas.
**Autor:** Erick Ilares

Este proyecto se centra en el análisis y categorización de sentimientos (positivo o negativo) de un conjunto de datos de 50,000 reseñas de películas. El objetivo ha sido construir un modelo de Machine Learning que pueda ser capaz de predecir el sentimiento de una reseña basándose en su texto.

### 🚀 Resumen del Proceso

El proyecto se dividió en las siguientes etapas clave:

1.  **Limpieza de Datos:** Preprocesamiento del texto de las reseñas para eliminar caracteres irrelevantes (HTML, puntuación, etc.) y convertirlas a un formato uniforme.
2.  **Análisis Exploratorio de Datos (EDA):** Se exploraron las palabras más comunes en las reseñas positivas y negativas para obtener **insights** sobre el tipo de lenguaje usado en cada sentimiento.
3.  **Vectorización de Texto:** Transformación del texto limpio a un formato numérico usando la técnica de **TF-IDF** (Term Frequency-Inverse Document Frequency), preparando los datos para el modelo.
4.  **Modelado de Machine Learning:** Entrenamiento y evaluación de un modelo de clasificación **Naive Bayes** para predecir el sentimiento de las reseñas.

### 📊 Resultados y Conclusiones

El modelo de clasificación **Naive Bayes** demostró ser efectivo en la tarea de análisis de sentimientos.

* **Precisión del modelo:** 85 %
* **Informe de Clasificación:**  
  
Métrica	Precision	Recall	F1-Score	Support
Negative	0.85	   0.85	   0.85   	4961
Positive	0.85   	0.85	   0.85	    5039
Accuracy			             0.85	    10000
Macro avg	0.85	  0.85	   0.85	    10000
Weighted avg	0.85	0.85	 0.85   	10000

**Conclusión:**
Los resultados obtenidos demuestran que el modelo es capaz de clasificar las reseñas con una **precisión** del 85 %, lo que lo convierte en una solución viable para predecir el sentimiento de nuevas reseñas de películas. Esto demuestra la capacidad de las técnicas de **NLP** (Procesamiento del Lenguaje Natural) para extraer información útil de datos no estructurados.

### 🔧 Tecnologías Utilizadas

* **Python**
* **Pandas:** Para la manipulación y limpieza de datos.
* **Scikit-learn:** Para la vectorización (TF-IDF), el modelado de Machine Learning (Naive Bayes) y la evaluación de resultados.
* **Matplotlib y WordCloud:** Para la visualización de datos (EDA).

