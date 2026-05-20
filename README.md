# ProyectoML

Repositorio de notebooks con casos de uso de machine learning aplicados a escenarios fintech.
Cada notebook resuelve un problema distinto usando datos sintéticos y técnicas comunes de análisis, modelado y evaluación.

## Notebooks incluidos

### 1. `01_prediccion_mora_logistic_regression.ipynb`
**Predicción de mora crediticia con Logistic Regression**

Este notebook simula un problema de riesgo crediticio: predecir si un cliente caerá en mora.

**Objetivo**
- Entrenar un modelo de clasificación binaria para predecir default crediticio.

**Tecnología de machine learning**
- `Logistic Regression`
- Preprocesamiento con `ColumnTransformer`
- `OneHotEncoder` para variables categóricas
- `StandardScaler` para variables numéricas

**Qué hace**
- Genera un dataset sintético de clientes.
- Realiza análisis exploratorio.
- Prepara las variables para entrenamiento.
- Entrena un modelo de clasificación binaria.
- Evalúa con:
  - Matriz de confusión
  - ROC-AUC
  - Precision
  - Recall
  - F1-score
- Interpreta el modelo mediante coeficientes.

---

### 2. `02_deteccion_fraude_random_forest.ipynb`
**Detección de fraude financiero con Random Forest**

Este notebook simula transacciones financieras y clasifica si una operación es fraudulenta o no.

**Objetivo**
- Entrenar un modelo de clasificación binaria para detección de fraude.

**Tecnología de machine learning**
- `Random Forest Classifier`
- Manejo de clases desbalanceadas con `class_weight="balanced"`
- Preprocesamiento con `ColumnTransformer`
- `OneHotEncoder` y `StandardScaler`

**Qué hace**
- Genera un dataset sintético de transacciones.
- Crea variables de comportamiento transaccional.
- Entrena un modelo de bosque aleatorio.
- Evalúa con:
  - Matriz de confusión
  - ROC-AUC
  - Precision
  - Recall
  - F1-score
- Muestra la importancia de variables para interpretación.

---

### 3. `03_segmentacion_clientes_kmeans.ipynb`
**Segmentación de clientes fintech con KMeans**

Este notebook agrupa clientes según su comportamiento financiero para descubrir segmentos de negocio.

**Objetivo**
- Construir segmentos de clientes como alto valor, endeudados, digitales activos o masivos.

**Tecnología de machine learning**
- `KMeans`
- Clustering no supervisado
- `StandardScaler` para normalización
- `PCA` para visualización en 2 dimensiones

**Qué hace**
- Genera un dataset sintético de clientes.
- Escala las variables.
- Usa el método del codo para elegir el número de clusters.
- Aplica KMeans para segmentar clientes.
- Resume cada cluster con estadísticas promedio.
- Visualiza los grupos con PCA.
- Asigna etiquetas interpretativas a los segmentos.

---

### 4. `04_forecast_finops_prophet_sarimax.ipynb`
**Forecast FinOps de costos cloud con modelo de serie de tiempo**

Este notebook simula costos diarios de infraestructura cloud y predice el gasto futuro.

**Objetivo**
- Construir un modelo de forecasting para estimar costos cloud.

**Tecnología de machine learning / series de tiempo**
- `SARIMAX` de `statsmodels`
- Modelado de series temporales
- Tendencia
- Estacionalidad semanal y mensual
- Medias móviles
- Métricas `MAE` y `RMSE`

**Qué hace**
- Genera una serie temporal sintética de 3 años.
- Incorpora tendencia, estacionalidad y eventos especiales.
- Grafica la evolución histórica.
- Entrena un modelo SARIMAX.
- Evalúa el pronóstico con métricas de error.
- Genera proyección futura y estimación mensual.

> Nota: en el notebook se menciona Prophet, pero la implementación actual usa `SARIMAX`.

---

### 5. `05_anomaly_detection_transactions_isolation_forest.ipynb`
**Detección de anomalías transaccionales con Isolation Forest**

Este notebook identifica transacciones anómalas sin usar etiquetas reales de fraude.

**Objetivo**
- Detectar transacciones fuera del patrón normal de comportamiento.

**Tecnología de machine learning**
- `Isolation Forest`
- Aprendizaje no supervisado
- `StandardScaler`
- Scoring de anomalías

**Qué hace**
- Genera transacciones normales y anómalas.
- Entrena un modelo `Isolation Forest`.
- Calcula puntajes de anomalía.
- Clasifica observaciones como normales o anómalas.
- Visualiza outliers por monto y distancia.
- Ordena registros por mayor nivel de anomalía.

---

## Resumen de tecnologías usadas

| Notebook | Problema | Técnica principal | Tipo de aprendizaje |
|---|---|---|---|
| `01_prediccion_mora_logistic_regression.ipynb` | Riesgo crediticio | Logistic Regression | Supervisado |
| `02_deteccion_fraude_random_forest.ipynb` | Fraude transaccional | Random Forest | Supervisado |
| `03_segmentacion_clientes_kmeans.ipynb` | Segmentación de clientes | KMeans | No supervisado |
| `04_forecast_finops_prophet_sarimax.ipynb` | Pronóstico de costos cloud | SARIMAX | Series de tiempo |
| `05_anomaly_detection_transactions_isolation_forest.ipynb` | Detección de anomalías | Isolation Forest | No supervisado |

## Requisitos generales

Los notebooks están desarrollados en Python 3.11 y usan librerías como:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `statsmodels`

## Cómo usar

1. Abre cualquiera de los notebooks en Jupyter, VS Code o PyCharm.
2. Ejecuta las celdas en orden.
3. Revisa los resultados, métricas y visualizaciones de cada caso.

## Estructura del proyecto

- `01_prediccion_mora_logistic_regression.ipynb`
- `02_deteccion_fraude_random_forest.ipynb`
- `03_segmentacion_clientes_kmeans.ipynb`
- `04_forecast_finops_prophet_sarimax.ipynb`
- `05_anomaly_detection_transactions_isolation_forest.ipynb`
- `data/`
- `models/`

## Objetivo del repositorio

Este proyecto funciona como portafolio de ejemplos de machine learning aplicados a fintech, cubriendo:

- clasificación
- clustering
- forecasting
- detección de anomalías
- evaluación de modelos
- interpretación básica de resultados

