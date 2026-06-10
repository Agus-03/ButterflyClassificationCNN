# 🦋 Clasificación Multiclase de Especies de Mariposas mediante CNN Propia

Este proyecto implementa, optimiza y evalúa el desarrollo iterativo de una **Red Neuronal Convolucional (CNN)** diseñada desde cero utilizando **TensorFlow y Keras**. El objetivo principal es clasificar imágenes de mariposas dentro de un abanico complejo de **75 especies independientes**, superando las limitaciones impuestas por la escasez crítica de datos nativos.

---

## 📌 Desafío del Dataset y Planteamiento del Problema

El proyecto se enfrenta a un escenario de alta complejidad en visión por computadora caracterizado por:
* **Gran Volumen de Clases:** 75 categorías morfológicas y biológicas distintas.
* **Escasez Crítica de Datos:** Entre 12 y 15 imágenes disponibles por clase.
* **Límite Probabilístico:** La probabilidad de acierto por azar puro es de apenas un **1.3%**.

Ante estas restricciones, cualquier arquitectura convencional tiende a memorizar el conjunto de entrenamiento de forma inmediata. El éxito de este proyecto radica en la aplicación ordenada de técnicas de **regularización de pesos, reestructuración geométrica y aumentación de datos** para forzar la capacidad de generalización de un modelo propio.

---

## 🚀 Evolución y Análisis de Resultados (Proceso de Ingeniería)

El desarrollo del clasificador se dividió en 4 etapas iterativas bien definidas, incrementando el rendimiento métrico de manera constante:

### 🔹 Etapa 1: Modelo Base (CNN Inicial)
* **Cambios Implementados:** Diseño de una arquitectura secuencial básica (2 bloques convolucionales sencillos), una capa de aplanamiento tradicional `Flatten()` y una capa densa de clasificación entrenada durante 20 épocas.
* **Justificación Técnica:** Establecer un punto de partida (*baseline*) empírico para medir la capacidad de aprendizaje lineal de la red ante las 75 clases.
* **Resultado y Diagnóstico:** Se alcanzó un **40.92% de Accuracy**. Las curvas de pérdida evidenciaron un **sobreajuste (overfitting) crítico e incontrolado**, donde el modelo memorizaba los datos de entrenamiento pero fallaba al validar.

### 🔹 Etapa 2: Estabilización y Regularización Básica
* **Cambios Implementados:** Incorporación de capas de `BatchNormalization`, penalización por regularización de pesos `L2` en la capa densa y configuración de los callbacks dinámicos `ReduceLROnPlateau` y `EarlyStopping`.
* **Justificación Técnica:** Controlar la varianza, estabilizar el flujo de los gradientes internos disminuyendo el desvanecimiento del error y mitigar las oscilaciones caóticas del optimizador ralentizando el aprendizaje dinámicamente si la pérdida se estancaba.
* **Resultado:** **60.62% de Accuracy** (+19.70% de mejora neta). El proceso se detuvo de forma óptima en la época 30 resguardando los mejores pesos antes de divergir.

### 🔹 Etapa 3: Reestructuración Arquitectónica Profunda
* **Cambios Implementados:** Inclusión de un tercer bloque convolucional profundo (escalando a 128 filtros), reducción de la tasa de aprendizaje inicial fija a `0.0001` (Adam) y sustitución radical de la capa `Flatten()` por un bloque de reducción espacial global **`GlobalAveragePooling2D()`**.
* **Justificación Técnica:** La capa `Flatten()` generaba un vector gigantesco que disparaba los parámetros a más de 33 millones, introduciendo un exceso de ruido. Al implementar *Global Average Pooling*, se promediaron las características espaciales, colapsando los millones de parámetros redundantes a solo unos miles y forzando a la red a extraer patrones abstractos generales (texturas y filamentos de las alas).
* **Resultado:** **69.15% de Accuracy** con un F1-Score balanceado de **69.09%**. Las curvas de *Train* y *Val* se estabilizaron de manera paralela con una brecha muy estrecha y saludable.

### 🔹 Etapa 4: Optimización Avanzada y Aumento de Datos (Modelo Definitivo)
* **Cambios Implementados:** Inserción de capas de Aumento de Datos en tiempo de ejecución (`DataAugmentation` con rotaciones, volteos horizontales/verticales, zooms y ajustes de contraste aleatorios) directamente como bloque de entrada de la red, junto con sintonización fina del callback `ReduceLROnPlateau`.
* **Justificación Técnica:** Expandir artificialmente el dataset en cada época para obligar al modelo a desarrollar **invarianza óptica**: reconocer las mariposas sin importar cambios de orientación, encuadre o iluminación de las fotografías de entrada.
* **Resultado Final:** **80.62% de Accuracy**, **82.22% de Precision** y **80.64% de F1-Score**. La pérdida de validación descendió de forma impecable y se aplanó por completo a partir de la época 25, erradicando el sobreajuste al 100%.

---

## 📊 Tabla Comparativa de Rendimiento

| Etapa de Desarrollo | Principales Modificaciones Técnicas | Accuracy | Precision | F1-Score | Estado del Overfitting |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Modelo Base** | Red simple, capa `Flatten`, 20 épocas básicas. | 40.92% | 42.46% | 39.98% | Muy Crítico / Incontrolado |
| **2. Regularización** | `BatchNormalization`, Regularización L2, Callbacks. | 60.62% | -- | ~60.10% | Parcialmente Mitigado |
| **3. Reestructuración**| Tercer Bloque (128 filtros), `GlobalAveragePooling2D`, Adam 1e-4. | 69.15% | 74.09% | 69.09% | Controlado / Brecha Sana |
| **4. Modelo Definitivo**| Capas de `DataAugmentation` + Callbacks dinámicos. | **80.62%**| **82.22%**| **80.64%**| **Completamente Erradicado** |

---

## 🛠️ Tecnologías Utilizadas

* **Python**
* **TensorFlow / Keras** (Diseño de la red y capas personalizadas)
* **NumPy & Pandas** (Procesamiento y manipulación de datos)
* **Matplotlib & Seaborn** (Visualización de las curvas de aprendizaje y métricas)

---

## 🧠 Conclusiones del Proyecto

El desarrollo de este proyecto convalida empíricamente que en el campo del Aprendizaje Profundo (*Deep Learning*), el éxito de un modelo ante problemas con escasez de datos no depende de aumentar el tamaño de la red de forma indiscriminada. 

A través de la reducción estratégica de parámetros geométricos redundantes (reemplazo de `Flatten` por `GlobalAveragePooling2D`), el uso de regularizadores dinámicos en el gradiente y la expansión sintética del dataset mediante técnicas de aumentación, se logró elevar la exactitud en casi **40 puntos porcentuales**. Esto demuestra la viabilidad de diseñar clasificadores multiclase robustos, puristas y altamente competitivos **completamente desde cero** sin la dependencia absoluta de arquitecturas preentrenadas externas.
