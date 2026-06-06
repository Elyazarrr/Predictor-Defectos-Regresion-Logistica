# Predictor de Defectos de Software (NASA Dataset)

## Descripción del Proyecto
Este repositorio contiene la implementación matemática "desde cero" de un modelo de **Regresión Logística Multivariable**. El sistema está diseñado para la inferencia predictiva en Arquitecturas de Software, actuando como un filtro de Calidad (QA) para bloquear despliegues de módulos altamente propensos a fallos.

El algoritmo fue entrenado utilizando métricas estáticas de código (Líneas de Código, Complejidad Ciclomática y Volumen de Halstead) correspondientes al *Software Defect Prediction Dataset* del NASA Metrics Data Program.

## Arquitectura Matemática (Sin cajas negras)
Para demostrar el dominio de los métodos predictivos, no se utilizaron librerías de Machine Learning de alto nivel (como scikit-learn). El núcleo matemático fue construido puramente con `numpy` e incluye:
* Estandarización de variables continuas mediante Z-Score.
* Función de activación **Sigmoide** para compresión probabilística.
* Optimización de pesos mediante **Descenso del Gradiente** minimizando la función de costo Log Loss (Entropía Cruzada Binaria).

## Requisitos del Entorno
* Python 3.x
* NumPy
* Pandas

Para instalar las dependencias, ejecutar en la terminal:

```bash
pip install numpy pandas
``` 

## Ejecución del Sistema
Para correr el entrenamiento y la simulación en vivo, ejecute el archivo principal:

```bash
python main.py
```

El sistema dividirá los datos (80% entrenamiento / 20% validación), ajustará los coeficientes algebraicos, imprimirá la precisión de la matriz de confusión y ejecutará una evaluación de riesgo en tiempo real sobre un módulo de prueba.

---
**Desarrollado por:** Dylan Elyazar Barahona Romero
**Materia:** Métodos Numéricos