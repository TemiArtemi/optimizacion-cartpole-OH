import numpy as np
import gymnasium as gym

class RedNeuronalCartPole:
    def __init__(self, input_size=4, hidden_size=8, output_size=2):
        """
        Inicializamos la arquitectura de la red neuronal.
        CartPole tiene 4 variables de observación y 2 acciones posibles.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inicialización aleatoria de pesos (W) y sesgos (b)
        # Usamos distribución normal estándar escalada para evitar valores extremos
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))
        
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros((1, self.output_size))

    def relu(self, x):
        """Función de activación no lineal para la capa oculta."""
        return np.maximum(0, x)

    def forward(self, x):
        """
        Paso hacia adelante (forward pass).
        Matemáticas puras: Multiplicación de matrices y suma de sesgos.
        """
        # Aseguramos que la entrada es un vector fila (1, 4)
        x = np.atleast_2d(x)
        
        # Capa oculta: Z1 = X * W1 + b1 -> A1 = ReLU(Z1)
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.relu(z1)
        
        # Capa de salida: Z2 = A1 * W2 + b2
        z2 = np.dot(a1, self.W2) + self.b2
        
        # Como CartPole requiere una acción discreta (0 o 1), 
        # devolvemos el índice de la neurona con el valor más alto (argmax)
        return np.argmax(z2)

    def get_pesos_como_cromosoma(self):
        """
        Convierte todas las matrices de la red en un único vector unidimensional.
        Este es el 'cromosoma' que el algoritmo genético cruzará y mutará.
        """
        vector = np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])
        return vector

    def set_pesos_desde_cromosoma(self, cromosoma):
        """
        Toma un vector unidimensional mutado por el algoritmo genético
        y reconstruye las matrices de pesos y sesgos de la red.
        """
        # Calculamos los índices de corte
        idx_w1 = self.input_size * self.hidden_size
        idx_b1 = idx_w1 + self.hidden_size
        idx_w2 = idx_b1 + (self.hidden_size * self.output_size)
        
        # Extraemos y redimensionamos
        self.W1 = cromosoma[0:idx_w1].reshape((self.input_size, self.hidden_size))
        self.b1 = cromosoma[idx_w1:idx_b1].reshape((1, self.hidden_size))
        self.W2 = cromosoma[idx_b1:idx_w2].reshape((self.hidden_size, self.output_size))
        self.b2 = cromosoma[idx_w2:].reshape((1, self.output_size))

def evaluar_fitness(cromosoma, env, render=False):
    """
    Evalúa la calidad de un cromosoma ejecutando un episodio de CartPole.
    El 'fitness' es la recompensa total (tiempo que el palo no se cae).
    """
    # Instanciamos la red y le inyectamos los pesos del cromosoma
    red = RedNeuronalCartPole()
    red.set_pesos_desde_cromosoma(cromosoma)
    
    estado, _ = env.reset()
    recompensa_total = 0
    terminado = falso = False
    
    while not (terminado or falso):
        if render:
            env.render()
            
        # La red decide la acción (0: izquierda, 1: derecha) basándose en el estado
        accion = red.forward(estado)
        
        # El entorno avanza un paso
        estado, recompensa, terminado, falso, _ = env.step(accion)
        recompensa_total += recompensa
        
    return recompensa_total

# --- Prueba rápida de funcionamiento ---
if __name__ == "__main__":
    # Creamos el entorno
    entorno = gym.make("CartPole-v1")
    
    # Creamos una red aleatoria y sacamos su cromosoma (su ADN)
    red_prueba = RedNeuronalCartPole()
    adn_aleatorio = red_prueba.get_pesos_como_cromosoma()
    
    print(f"Tamaño del cromosoma (total de pesos): {len(adn_aleatorio)}")
    
    # Evaluamos qué tan buena es esta red aleatoria
    puntuacion = evaluar_fitness(adn_aleatorio, entorno)
    print(f"Fitness de la red aleatoria: {puntuacion} puntos")
    
    entorno.close()