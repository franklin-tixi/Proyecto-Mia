# Sistema de Inteligencia Artificial Explicable para la Mejora en la Precisión de Proyecciones Financieras del Portafolio de Clientes de la Consultora

## Descripción
El desarrollo de este proyecto responde a la necesidad de mejorar la calidad, consistencia y confiabilidad de las proyecciones financieras en la consultora. Desde una perspectiva organizacional, la implementación de un sistema basado en inteligencia artificial permitirá optimizar los tiempos de análisis, reducir la dependencia del criterio humano y estandarizar los procesos de evaluación financiera.
Desde el punto de vista tecnológico, el proyecto integra modelos de machine learning con técnicas de inteligencia artificial explicable (XAI)(Mandava et al., 2025), lo cual es especialmente relevante en entornos financieros donde la transparencia y trazabilidad de los resultados son fundamentales.
En el ámbito académico, el proyecto contribuye al análisis y validación de modelos predictivos aplicados a datos financieros reales, permitiendo evaluar su desempeño y su capacidad de interpretación.

## 💡 Solución propuesta
Se implementó una solucion que permite:
- Cargar y limpiar los datos históricos
- Agrupar el saldo por año y período
- Construir una variable temporal continua
- Entrenar múltiples modelos predictivos
- Exportar resultados y evidencias

## Modelos utilizados
- Regresión Lineal
- Random Forest
- Red Neuronal (MLPRegressor)

## Pipeline
1. Preprocesamiento de datos
   - Lectura del archivo Excel
   - Limpieza de datos
   - Conversión de variables a formato numérico
   - Agrupación por año y período

2. Entrenamiento de modelos
   - Entrenamiento de modelos de regresión
   - Ajuste de modelos con los datos históricos

3. Proyección
   - Generación de períodos futuros
   - Predicción con cada modelo
   - Construcción del dataset de resultados

4. Visualización
  - Comparación entre datos reales y predicciones
  - Generación de gráficos
 
## Requisitos técnicos
El proyecto requiere Python 3.10 o superior y las siguientes librerías: pandas, numpy, matplotlib, scikit-learn, openpyxl y joblib.

## Instrucciones de ejecución
El proyecto debe ejecutarse en el siguiente orden: primero python scripts/preprocess.py para generar el dataset procesado, luego python scripts/train.py para entrenar los modelos y finalmente python scripts/forecast.py

## Control de versiones
El proyecto fue gestionado mediante GitHub, manteniendo una estructura organizada de carpetas, código modular y documentación clara.

## Ejecución

```bash
python scripts/preprocess.py
python scripts/train.py
python scripts/forecast.py
python scripts/grafico.py
