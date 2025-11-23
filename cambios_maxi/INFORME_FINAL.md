# Informe Final: Optimización del Modelo de Predicción de Resiliencia

## 1. Introducción
El presente trabajo se centró en la mejora y optimización de un modelo de Inteligencia Artificial preexistente, diseñado para predecir el puntaje de resiliencia a partir de datos de encuestas. El objetivo principal fue reducir el Error Absoluto Medio (MAE) del modelo base, que se situaba inicialmente en 3.38.

## 2. Metodología y Mejoras Implementadas

Se realizaron una serie de modificaciones incrementales sobre el modelo original para mejorar su capacidad de generalización y precisión.

### 2.1 Preprocesamiento de Datos
*   **Escalado de Características (StandardScaler)**: Se identificó que las variables de entrada, al ser categóricas codificadas numéricamente (Label Encoding), poseían diferentes escalas. Se implementó la normalización de todas las características para tener media 0 y desviación estándar 1.
    *   **Resultado**: Esta modificación facilitó la convergencia de la red neuronal y proporcionó la mejora individual más significativa en el rendimiento.

### 2.2 Modificaciones en la Arquitectura y Entrenamiento
*   **Función de Pérdida L1Loss**: Se sustituyó la función de pérdida original por `L1Loss`. Dado que la métrica de evaluación es el MAE, optimizar directamente el error absoluto resultó ser más efectivo que optimizar el error cuadrático medio (MSE).
*   **Batch Normalization**: Se añadieron capas de normalización por lotes después de cada capa lineal (excepto la de salida). Esto estabilizó el proceso de entrenamiento y permitió el uso de tasas de aprendizaje más altas.
*   **Learning Rate Scheduler**: Se implementó `ReduceLROnPlateau` para reducir dinámicamente la tasa de aprendizaje cuando la pérdida de validación dejaba de mejorar, permitiendo un ajuste fino de los pesos en las etapas finales.

### 2.3 Estrategia de Validación y Ensamble
*   **K-Fold Cross-Validation (5 Folds)**: Se adoptó una estrategia de validación cruzada para evaluar el modelo en 5 subconjuntos diferentes de datos.
*   **Ensamble de Modelos**: En lugar de seleccionar un único mejor modelo, se conservaron los 5 modelos generados durante la validación cruzada. La predicción final se calcula como el promedio de las salidas de estos 5 modelos.
    *   **Resultado**: Esta técnica redujo la varianza de las predicciones y mejoró la robustez del sistema frente a datos no vistos.

## 3. Resultados Obtenidos

Las mejoras implementadas permitieron reducir el error de predicción de manera consistente.

| Métrica | Modelo Base | Modelo Final Optimizado | Mejora |
| :--- | :--- | :--- | :--- |
| **MAE** | 3.38 | **2.998** | **11.3%** |
| **R²** | 0.40 | **0.488** | **+0.088** |

El modelo final (`best_model.pth`) consiste en el ensamble de las 5 redes neuronales entrenadas.

## 4. Experimentos Adicionales Realizados

Durante el proceso de optimización, se evaluaron otras estrategias que finalmente fueron descartadas por no superar el rendimiento del enfoque base optimizado.

### 4.1 Entity Embeddings
*   **Hipótesis**: Se intentó sustituir el Label Encoding por capas de Embedding aprendibles para las 31 variables categóricas, con el fin de capturar relaciones semánticas entre categorías.
*   **Resultado**: MAE 3.39.
*   **Análisis**: El modelo sufrió de sobreajuste (overfitting). El tamaño del dataset (3,457 muestras) resultó insuficiente para entrenar eficazmente la gran cantidad de parámetros adicionales que introducen los embeddings.

### 4.2 One-Hot Encoding
*   **Hipótesis**: Se evaluó el uso de One-Hot Encoding para eliminar la ordinalidad implícita del Label Encoding.
*   **Resultado**: MAE 3.34.
*   **Análisis**: Esta codificación aumentó la dimensionalidad de entrada de 31 a 87 características. El aumento en la dispersión de los datos (sparsity) dificultó el aprendizaje en comparación con la representación más compacta del Label Encoding escalado.

### 4.3 Ingeniería de Características (Feature Engineering)
*   **Hipótesis**: Se crearon manualmente variables de interacción (ej. suma de estresores, interacción social-salud).
*   **Resultado**: MAE 3.44.
*   **Análisis**: Las nuevas variables introdujeron multicolinealidad y ruido, degradando el rendimiento del modelo en lugar de mejorarlo.

## 5. Conclusión

La optimización del modelo se logró principalmente mediante la estandarización de los datos de entrada y la implementación de una estrategia de ensamble robusta (K-Fold). Las técnicas más complejas como Embeddings o ingeniería de características manual no aportaron valor adicional debido a las limitaciones en el tamaño del dataset. El modelo final cumple con el objetivo de reducir el error absoluto medio por debajo de 3.0.
