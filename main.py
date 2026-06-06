import pandas as pd
import numpy as np
from modelo_matematico import RegresionLogisticaDesdeCero

print("\n[INFO] Iniciando Sistema de Predicción de QA (NASA Dataset)...")

try:
    # Intenta leer el dataset real si ya lo tienes en la carpeta
    df = pd.read_csv('SoftwareDefectDataset.csv')
    print("[INFO] Dataset real 'SoftwareDefectDataset.csv' cargado exitosamente.")
except FileNotFoundError:
    # Sistema de respaldo: Genera datos sintéticos estructuralmente idénticos
    # para que puedas probar el código antes de descargar el CSV oficial.
    print("[WARN] No se encontró 'SoftwareDefectDataset.csv'. Generando datos de simulación para la demo...")
    np.random.seed(42)
    datos = {
        'LOC': np.random.randint(10, 1500, 300),     # Líneas de código
        'CYCLO': np.random.randint(1, 50, 300),      # Complejidad Ciclomática
        'VOLUME': np.random.uniform(100, 8000, 300)  # Volumen de Halstead
    }
    df = pd.DataFrame(datos)
    # Lógica de simulación: a mayor tamaño y complejidad, mayor riesgo de defecto
    probabilidad_riesgo = (df['LOC']/1500 + df['CYCLO']/50) / 2
    df['DEFECT_LABEL'] = [1 if p > 0.45 else 0 for p in probabilidad_riesgo]

# Separar las métricas continuas (X) de la etiqueta categórica (y)
columnas_entrada = ['LOC', 'CYCLO', 'VOLUME']
X = df[columnas_entrada].values
y = df['DEFECT_LABEL'].values

# OBLIGATORIO: Si no escalamos los datos, el Descenso del Gradiente colapsa
# porque la métrica VOLUME (miles) aplastaría a CYCLO (decenas).
def estandarizar(matriz):
    media = np.mean(matriz, axis=0)
    desviacion = np.std(matriz, axis=0)
    return (matriz - media) / desviacion

X_escalado = estandarizar(X)

# División del Dataset: 80% Entrenamiento / 20% Prueba 
indices = np.arange(X_escalado.shape[0])
np.random.shuffle(indices) # Mezclar los datos aleatoriamente
corte = int(0.8 * X_escalado.shape[0])

X_train, X_test = X_escalado[indices[:corte]], X_escalado[indices[corte:]]
y_train, y_test = y[indices[:corte]], y[indices[corte:]]

print(f"[INFO] Datos divididos: {X_train.shape[0]} para entrenamiento, {X_test.shape[0]} para validación.")

print("\n[INFO] Inicializando Descenso del Gradiente...")
# Instanciamos nuestra clase matemática creada desde cero
modelo = RegresionLogisticaDesdeCero(tasa_aprendizaje=0.1, iteraciones=2000)

# El motor matemático comienza a ajustar los pesos 
modelo.entrenar(X_train, y_train)
print("[INFO] Entrenamiento matemático completado.")

# Validación de la precisión contra datos nunca antes vistos (El 20% restante)
predicciones = modelo.predecir(X_test)

# Calcular precisión matemática manualmente
aciertos = np.sum(predicciones == y_test)
precision = (aciertos / len(y_test)) * 100

print("\n==========================================================")
print(f"🎯 RESULTADOS DE LA PREDICCIÓN (REGRESIÓN LOGÍSTICA)")
print("==========================================================")
print(f"Precisión del Modelo: {precision:.2f}% de aciertos.")
print("==========================================================")

print("\n--- TEST EN VIVO: Evaluando un nuevo módulo de código ---")
# Supongamos que acabamos de escribir un código muy denso (LOC=1200, CYCLO=45, VOLUME=7000)
nuevo_modulo = np.array([[1200, 45, 7000]])

# IMPORTANTE: Debemos estandarizar este nuevo dato usando la misma escala del entrenamiento
media_train = np.mean(X, axis=0)
desv_train = np.std(X, axis=0)
nuevo_modulo_escalado = (nuevo_modulo - media_train) / desv_train

# Pedimos la probabilidad exacta al modelo
probabilidad = modelo.predecir_probabilidad(nuevo_modulo_escalado)[0]
decision = modelo.predecir(nuevo_modulo_escalado)[0]

print(f"Métricas del Módulo -> LOC: 1200, CYCLO: 45, VOLUME: 7000")
print(f"Cálculo Sigmoide   -> {probabilidad * 100:.2f}% de probabilidad de fallo.")
if decision == 1:
    print("❌ DICTAMEN DEL SISTEMA: ALERTA DE DEFECTO. BLOQUEAR DESPLIEGUE A PRODUCCIÓN.")
else:
    print("✅ DICTAMEN DEL SISTEMA: MÓDULO LIMPIO. APROBADO PARA PRODUCCIÓN.")