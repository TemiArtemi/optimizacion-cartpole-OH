# AGENTS.MD - Guía de Operación y Reglas de Desarrollo

Este documento define el comportamiento, las normas de codificación, la arquitectura y el flujo de trabajo obligatorio para cualquier agente de inteligencia artificial que colabore en este repositorio.

---

## 1. Contexto Básico del Proyecto

- **Asignatura:** Optimización y Heurística (Grado en Ciencia e Ingeniería de Datos - ULPGC).
- **Proyecto:** Optimización de redes neuronales con algoritmos genéticos (Neuroevolución).
- **Objetivo Principal:** Desarrollar desde cero algoritmos evolutivos aplicados a redes neuronales en dos vertientes:
  1. Optimizar directamente los pesos de una red neuronal pequeña sin usar gradientes en un entorno de control donde la función objetivo no es diferenciable (*CartPole-v1* de `Gymnasium`), comparando su comportamiento con el enfoque clásico.
  2. Optimizar la arquitectura y los hiperparámetros de una red neuronal mayor entrenada mediante descenso por gradiente sobre un conjunto de datos tabulares real (*UCI Machine Learning Repository*).
- **Entorno de ejecución:** Python local (ejecución en CPU con técnicas de paralelización).

---

## 2. Reglas de Código: Idioma, Comentarios, Clean Code y SOLID

### 2.1 REGLA DE IDIOMA: Todo el Código en Inglés

> ⚠️ **REGLA CRÍTICA:** **Todo artefacto de código debe estar redactado íntegramente en inglés.**

Esto aplica de forma obligatoria e inequívoca a:
1. **Identificadores:** Nombres de módulos, paquetes, clases, métodos, funciones, variables, parámetros y constantes.
2. **Rutas y ficheros:** Nombres de archivos fuente, scripts, carpetas y artefactos generados.
3. **Claves de diccionarios, atributos y mensajes por consola/errores.**

### 2.2 Idioma de la Documentación e Interacción: Español

- **Documentación externa y entregables:** Todo archivo `.md` (`README.md`, memoria, registros de experimentos), así como los títulos y etiquetas de las gráficas para la memoria académica, deben redactarse en **español**.
- **Conversación con el usuario:** Toda explicación técnica, propuesta y debate en el chat se mantendrá estrictamente en **español**.

### 2.3 REGLA SUPREMA: Prohibición de Comentarios y Restricción de Docstrings

> ⚠️ **REGLA CRÍTICA E INQUEBRANTABLE:**  
> **ESTÁ ESTRICTAMENTE PROHIBIDO INCLUIR COMENTARIOS EN LÍNEA (`#`). SOLO SE PERMITEN DOCSTRINGS MUY BREVES.**

1. **Cero comentarios con `#`:** Prohibido el uso de `#` dentro de cualquier archivo `.py` o notebook `.ipynb`.
2. **Docstrings mínimos:** Se permite única y exclusivamente un *docstring* de **una sola línea** (máximo dos si es estrictamente necesario) al inicio de cada clase o función para describir su propósito general (`"""Evaluates a chromosome in the CartPole environment and returns total reward."""`). Prohibido incluir bloques largos de parámetros o explicaciones matemáticas en los docstrings.
3. **Código autodocumentado:** Toda la claridad debe provenir del diseño del software. Las explicaciones matemáticas, algorítmicas y de diseño se darán siempre en el chat al usuario o en documentos `.md`, nunca ensuciando el código fuente.

### 2.4 Clean Code y Principios SOLID

- **Principios SOLID:** Énfasis estricto en el Principio de Responsabilidad Única (SRP) y Principio Abierto/Cerrado (OCP). La lógica de la red neuronal, el entorno de simulación, los operadores genéticos (selección, cruce, mutación) y el motor evolutivo deben estar completamente desacoplados.
- **Tipado Estricto (*Type Hints*):** Toda función y método debe declarar explícitamente los tipos de sus argumentos y su valor de retorno usando el módulo `typing` y `numpy.typing` (`np.ndarray`, `int`, `float`, `Tuple[np.ndarray, np.ndarray]`, `Callable`).
- **Funciones Puras:** Los operadores genéticos deben implementarse como funciones puras o clases sin efectos secundarios ocultos sobre los arrays originales (evitar mutaciones *in-place* no deseadas).

---

## 3. Requisitos Técnicos Específicos del Proyecto

1. **Implementación Propia del Núcleo:** El bucle del algoritmo genético, la codificación/decodificación del cromosoma y los operadores evolutivos deben programarse desde cero. No se permite usar librerías de algoritmos genéticos prefabricadas (como DEAP o PyGAD) para el motor principal, aunque puedan usarse como referencia comparativa si se requiere.
2. **Fase 1 en NumPy Puro:** La red neuronal para *CartPole-v1* (paso hacia adelante, activaciones, aplanado y reconstrucción de matrices de pesos) debe implementarse exclusivamente con álgebra matricial vectorizada en `NumPy`, sin utilizar `PyTorch` ni `TensorFlow`.
3. **Fase 2 con Gradientes (PyTorch / Scikit-learn):** Para la optimización de arquitecturas e hiperparámetros sobre datos de UCI, el algoritmo genético actuará como optimizador externo (caja negra) sobre redes entrenadas con descenso por gradiente.
4. **Operadores Genéticos Adecuados:** Diseño y comparativa de operadores específicos para pesos continuos y arquitecturas discretas (selección por torneo/ruleta, cruce de un punto, cruce uniforme, cruce aritmético/BLX-alpha, mutación gaussiana adaptativa).
5. **Paralelización y Coste Computacional:** La evaluación del *fitness* de la población es el cuello de botella. Se debe implementar evaluación paralela en CPU (usando `multiprocessing` o `concurrent.futures`) y medir rigurosamente la ganancia en tiempos de ejecución (*speedup*).
6. **Rigor Experimental y Reproducibilidad:**
   - Comparar siempre contra una línea base (*baseline* de búsqueda aleatoria o retropropagación).
   - Dado el carácter estocástico de las metaheurísticas, ejecutar cada experimento múltiples veces con distintas semillas (`seeds`) fijadas, reportando media, desviación típica y curvas de convergencia.

---

## 4. Estructura Objetivo del Repositorio (EJEMPLO: puede variar)

El código deberá organizarse siguiendo una arquitectura modular limpia:

    ├── data/                      
    ├── results/                   
    │   ├── figures/               
    │   └── logs/                  
    ├── src/                       
    │   ├── networks/              
    │   │   ├── numpy_mlp.py       
    │   │   └── torch_mlp.py       
    │   ├── genetic/               
    │   │   ├── operators.py       
    │   │   ├── selection.py       
    │   │   └── engine.py          
    │   ├── environments/          
    │   │   └── cartpole_eval.py   
    │   └── utils/                 
    │       ├── metrics.py         
    │       └── plotting.py        
    ├── experiments/               
    ├── requirements.txt           
    ├── AGENTS.MD                  
    └── README.md                  

---

## 5. Modo de Trabajo y Flujo de Interacción

El agente tiene **prohibido** generar múltiples archivos o grandes bloques de código de forma impulsiva. Debe trabajar de forma iterativa y guiada:

1. **Planteamiento y Planificación:** Antes de escribir o modificar código, el agente debe exponer claramente qué se va a hacer en ese paso y cómo encaja en el plan general.
2. **Preguntas Aclaratorias:** Si existen varias alternativas matemáticas o de diseño (p. ej., qué método de selección implementar primero, tamaño de población, criterio de parada), el agente debe plantear las opciones al usuario y esperar su decisión.
3. **Explicación Teórica y Técnica Completa:** Cada vez que el agente entregue código, debe explicar detalladamente en el chat (en español) la formulación matemática aplicada, el porqué de las decisiones de diseño (SOLID/Clean Code) y cómo verificar que funciona.
4. **Plan de Desarrollo por Fases**

---

## 6. Política de Control de Versiones (Git)

Se aplicará de forma estricta el estándar **Conventional Commits**:

- **Idioma y formato:** Todos los mensajes de commit deben escribirse **estrictamente en inglés y en minúsculas**.
- **Prefijos permitidos:**
  - `feat:` para nuevas funcionalidades o algoritmos (ej. `feat: add tournament selection and elitism to genetic engine`).
  - `fix:` para corrección de errores o bugs (ej. `fix: prevent index zero in single point crossover`).
  - `refactor:` para cambios de estructura, limpieza o traducción sin alterar la lógica (ej. `refactor: migrate base neural network to english and add type hints`).
  - `docs:` para cambios en documentación, `README.md` o informes (ej. `docs: add experimental setup to readme`).
  - `chore:` para configuración de entorno, `.gitignore` o `requirements.txt` (ej. `chore: add matplotlib and scipy to requirements`).
- **Atomicidad:** Los commits deben ser atómicos, ordenados secuencialmente y reflejar pasos lógicos individuales del desarrollo. El agente debe indicar al usuario el comando `git add` y `git commit` exacto al finalizar y validar cada subtarea.