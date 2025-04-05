# Documentación del Proyecto de Análisis de Incendios

Este proyecto integral tiene como objetivo analizar, predecir y clasificar datos sobre incendios forestales en México utilizando técnicas avanzadas de machine learning. El proyecto está organizado en cuatro módulos principales: Pre-Procesamiento, Regresión, Clustering y Clasificación, cada uno con funcionalidades específicas y complementarias.

## Estructura del Proyecto

```
├── Pre-Procesamiento/
│   ├── prepararDatos.py
│   └── proyectoBDLimpia.py
├── Regresion/
│   ├── BD.csv
│   ├── humedad-mexico.csv
│   ├── model.py
│   └── temperaturas-mexico.csv
├── Clusterizacion/
│   ├── BD.csv
│   ├── model_backend.py
│   └── model_ui.py
└── Clasificacion/
    ├── BD_IncendiosSNIF_2015-2022.csv
    ├── dataset_balanceado.csv
    ├── nuevo_dataset.csv
    ├── Modelo_Clasificacion.py
    ├── Modelo_Clasificacion_2.py
    └── PruebaClassification.ipynb
```

## 1. Módulo de Pre-Procesamiento

Este módulo se encarga de preparar los datos para su análisis posterior, limpiando y transformando el conjunto de datos original.

### Archivos:
- **prepararDatos.py**: Script para el procesamiento inicial de los datos.
- **proyectoBDLimpia.py**: Script que genera una base de datos limpia para su uso en los demás módulos.

### Funcionalidades:
- Limpieza de valores nulos y atípicos
- Transformación de variables
- Normalización de formatos (fechas, coordenadas, etc.)
- Validación de la integridad de los datos

## 2. Módulo de Regresión

Este módulo implementa un modelo de predicción para la duración de incendios utilizando XGBoost, complementado con información climática y geográfica.

### Archivos:
- **BD.csv**: Datos históricos de incendios.
- **temperaturas-mexico.csv**: Temperaturas promedio mensuales por estado.
- **humedad-mexico.csv**: Valores de humedad promedio mensuales por estado.
- **model.py**: Implementación del modelo de regresión.

### Funcionalidades:
1. **Carga y Procesamiento de Datos**:
   - Carga de datos históricos y variables climáticas
   - Generación de clusters geográficos mediante KMeans
   - Procesamiento de fechas y extracción de características temporales
   - Limpieza y normalización de datos textuales

2. **Ingeniería de Características**:
   - Asignación de variables climáticas (temperatura y humedad)
   - Transformación logarítmica de la variable objetivo
   - Selección de características relevantes

3. **Modelado Predictivo**:
   - Pipeline integrado con preprocesamiento (escalado y codificación)
   - Modelo XGBoost con hiperparámetros optimizados
   - Evaluación mediante MSE y R²
   - Almacenamiento del modelo entrenado

### Requisitos:
- pandas
- numpy
- scikit-learn
- xgboost
- joblib

### Ejecución:
```bash
python model.py
```

### Consideraciones:
- El modelo aplica transformación logarítmica para normalizar la variable objetivo
- Los hiperparámetros fueron optimizados previamente
- Se incluyen factores climáticos, geográficos y temporales como predictores

## 3. Módulo de Clustering

Este módulo permite analizar y visualizar datos de incendios mediante técnicas de clustering y una interfaz gráfica interactiva.

### Archivos:
- **BD.csv**: Datos históricos de incendios.
- **model_backend.py**: Lógica de procesamiento, clustering y generación de matrices de riesgo.
- **model_ui.py**: Interfaz gráfica utilizando Tkinter y matplotlib.

### Funcionalidades:
1. **Procesamiento de Datos**:
   - Carga y limpieza de datos históricos
   - Generación de clusters regionales e individuales
   - Creación de matrices de riesgo

2. **Visualización Interactiva**:
   - Selección dinámica de años (2015-2023)
   - Visualización de clusters en mapas
   - Matrices de riesgo interactivas
   - Resúmenes estadísticos por región y ecosistema
   - Análisis histórico acumulado

### Requisitos:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pillow
- Tkinter

### Ejecución:
```bash
python model_ui.py
```

### Interfaz de Usuario:
- Navegación entre años mediante botones
- Pestañas para diferentes visualizaciones
- Panel lateral con leyendas explicativas
- Análisis histórico completo

## 4. Módulo de Clasificación

Este módulo implementa modelos de clasificación para categorizar y predecir tipos de incendios forestales.

### Archivos:
- **BD_IncendiosSNIF_2015-2022.csv**: Datos históricos de incendios.
- **dataset_balanceado.csv**: Dataset procesado con clases balanceadas.
- **nuevo_dataset.csv**: Dataset transformado para el modelado.
- **Modelo_Clasificacion.py**: Implementación del primer modelo de clasificación.
- **Modelo_Clasificacion_2.py**: Implementación mejorada del modelo de clasificación.
- **PruebaClassification.ipynb**: Notebook para pruebas y experimentación.

### Funcionalidades:
1. **Procesamiento de Datos**:
   - Carga y limpieza del conjunto de datos
   - Balanceo de clases para mejorar la representatividad
   - Selección y transformación de características

2. **Modelado de Clasificación**:
   - Implementación de diversos algoritmos (Random Forest, SVM, etc.)
   - Tuning de hiperparámetros
   - Validación cruzada y evaluación de modelos

3. **Evaluación y Selección de Modelos**:
   - Métricas de evaluación (Precisión, Recall, F1-Score)
   - Matrices de confusión
   - Curvas ROC y AUC

### Requisitos:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### Ejecución:
```bash
python Modelo_Clasificacion_2.py
```

## Integración de Módulos

El proyecto está diseñado para que los diferentes módulos puedan trabajar de manera complementaria:

1. El módulo de **Pre-Procesamiento** genera los conjuntos de datos limpios.
2. El módulo de **Regresión** utiliza estos datos para predecir la duración de incendios.
3. El módulo de **Clustering** agrupa los incendios según sus características y permite visualizaciones.
4. El módulo de **Clasificación** categoriza los tipos de incendios para análisis predictivo.

## Requisitos Generales del Proyecto

### Dependencias:
- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- pillow
- Tkinter
- joblib

### Instalación de Dependencias:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn pillow joblib
```

## Uso Recomendado

1. Ejecutar primero los scripts de pre-procesamiento para generar los datasets limpios.
2. Utilizar la aplicación de clustering para explorar patrones en los datos y generar visualizaciones.
3. Entrenar el modelo de regresión para predecir la duración de incendios.
4. Implementar los modelos de clasificación para categorizar nuevos incendios.

