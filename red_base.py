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


# ... (aquí arriba está tu clase RedNeuronalCartPole y evaluar_fitness) ...

def cruce_un_punto(padre1, padre2):
    """
    Realiza un cruce de un punto entre dos vectores (cromosomas) del mismo tamaño.
    Elige un índice aleatorio y combina las secciones de ambos padres.
    """
    punto_corte = np.random.randint(1, len(padre1))
    hijo1 = np.concatenate([padre1[:punto_corte], padre2[punto_corte:]])
    hijo2 = np.concatenate([padre2[:punto_corte], padre1[punto_corte:]])
    return hijo1, hijo2

def mutacion_gaussiana(cromosoma, prob_mutacion=0.1, media=0.0, desviacion=0.1):
    """
    Añade ruido gaussiano a cada elemento del vector con una probabilidad dada.
    """
    cromosoma_mutado = np.copy(cromosoma)
    probabilidades = np.random.rand(len(cromosoma_mutado))
    mutar_mask = probabilidades < prob_mutacion
    ruido = np.random.normal(media, desviacion, size=np.sum(mutar_mask))
    cromosoma_mutado[mutar_mask] += ruido
    return cromosoma_mutado

# --- Prueba rápida de funcionamiento ---
if __name__ == "__main__":
    entorno = gym.make("CartPole-v1")
    
    # Creamos dos redes "padre"
    red1 = RedNeuronalCartPole()
    red2 = RedNeuronalCartPole()
    
    adn_padre1 = red1.get_pesos_como_cromosoma()
    adn_padre2 = red2.get_pesos_como_cromosoma()
    
    # Probamos el cruce y la mutación
    adn_hijo1, adn_hijo2 = cruce_un_punto(adn_padre1, adn_padre2)
    adn_hijo1_mutado = mutacion_gaussiana(adn_hijo1)
    
    print(f"Tamaño padre: {len(adn_padre1)}")
    print(f"Tamaño hijo mutado: {len(adn_hijo1_mutado)}")
    print("¡Cruce y mutación ejecutados correctamente!")
    
    entorno.close()