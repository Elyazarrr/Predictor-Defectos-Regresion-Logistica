import numpy as np

class RegresionLogisticaDesdeCero:
    def __init__(self, tasa_aprendizaje=0.01, iteraciones=1000):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.iteraciones = iteraciones
        self.pesos = None
        self.sesgo = None

    def _sigmoide(self, z):
        # np.clip evita que números gigantes colapsen la función exponencial
        z = np.clip(z, -250, 250) 
        return 1 / (1 + np.exp(-z))

    def entrenar(self, X, y):
        # m = cantidad de registros (ej. 1000 módulos de la NASA)
        # n = cantidad de características (ej. 3 métricas)
        m, n = X.shape
        
        # 1. Inicializar los pesos (betas) y el sesgo en cero
        self.pesos = np.zeros(n)
        self.sesgo = 0

        # 2. Bucle de optimización (Descenso del Gradiente)
        for i in range(self.iteraciones):
            # Calcular la ecuación lineal: z = X*w + b
            modelo_lineal = np.dot(X, self.pesos) + self.sesgo
            
            # Pasar la línea por la curva sigmoide para obtener predicciones (probabilidades)
            y_predicho = self._sigmoide(modelo_lineal)

            # Calcular los gradientes (Derivadas de la función de costo Log Loss)
            dw = (1 / m) * np.dot(X.T, (y_predicho - y))
            db = (1 / m) * np.sum(y_predicho - y)

            # Actualizar los pesos moviéndonos en contra del gradiente
            self.pesos -= self.tasa_aprendizaje * dw
            self.sesgo -= self.tasa_aprendizaje * db

    def predecir_probabilidad(self, X):

        modelo_lineal = np.dot(X, self.pesos) + self.sesgo
        return self._sigmoide(modelo_lineal)

    def predecir(self, X, umbral=0.5):
        
        probabilidades = self.predecir_probabilidad(X)
        clases_clasificadas = [1 if prob > umbral else 0 for prob in probabilidades]
        return np.array(clases_clasificadas)