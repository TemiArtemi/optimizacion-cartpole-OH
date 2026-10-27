## OPTIMIZACIÓN Y HEURÍSTICA

## Propuestas de proyectos de la asignatura

Grado en Ciencia e Ingeniería de Datos · Escuela de Ingeniería Informática · ULPGC · Curso 2026-2027

Resumen. Cada estudiante (o grupo) elige un problema, investiga cómo resolverlo con técnicas de optimización y heurística, programa los métodos, los prueba con datos reales y lo documenta en una memoria. Opcionalmente, puede hacer una presentación para subir nota. Se puede y se recomienda usar la IA como herramienta de trabajo.

## 1. Objetivo del proyecto

El proyecto es la parte de la asignatura en la que se aplican conceptos vistos en clase a un problema concreto. Se trata de ir más allá de los ejercicios: entender a fondo un problema real o un método de optimización, investigar qué se ha hecho en la literatura, implementar soluciones y evaluarlas de forma rigurosa.

En concreto, en el proyecto hay que:

- 1. Investigar un problema concreto y entender por qué es un problema de optimización: qué se decide, qué se optimiza y qué restricciones hay.

- 2. Estudiar qué técnicas de optimización y heurística son adecuadas para resolverlo y por qué.

- 3. Programar los métodos de optimización (no basta con llamar a una librería: el núcleo del método debe ser una implementación propia, aunque se utilicen librerías para comparar).

- 4. Utilizar datos reales para comprobar que las soluciones funcionan, y compararlas con alternativas razonables.

- 5. Documentar de forma clara el problema, los métodos, los experimentos y las conclusiones.

Como se permite usar la IA para desarrollar los proyectos, el nivel esperado es alto. La IA permite abordar métodos más avanzados, más experimentos y problemas más realistas de lo que sería posible sin ella. Por eso la mayoría de las propuestas incluyen objetivos ambiciosos y un apartado "Para ir más allá".

## 2. Qué hay que entregar

| Entregable | Contenido | Carácter |
| --- | --- | --- |
| Memoria del proyecto | Documento que explique el problema, el estado del arte, la formulación, los métodos, la implementación, los experimentos, los resultados y las conclusiones (ver apartado 3). | Obligatorio |
| Código fuente | Implementación de los métodos y de los experimentos, con un README que explique cómo instalar y ejecutar todo y un fichero de dependencias (requirements.txt o similar). Se recomienda un repositorio Git. | Obligatorio |
| Datos | Los datos utilizados o, si son grandes, los scripts para descargarlos y preprocesarlos, indicando siempre su origen y licencia. | Obligatorio |


| Entregable | Contenido | Carácter |
| --- | --- | --- |
|   | Apartado sobre el uso de IA Descripción de cómo se ha utilizado la IA en el proyecto. | Obligatorio |
| Presentación | Exposición del proyecto con demostración de resultados. | Opcional |

## 3. Estructura recomendada de la memoria

La memoria debe poder leerla alguien que no conoce el proyecto y entender qué problema se ha resuelto, cómo y con qué resultados. Se recomienda la siguiente estructura, que puede adaptarse a cada proyecto:

- 1. Título del proyecto y Nombre de los autores

- 2. Resumen del proyecto. En menos de 300 palabras, realizar un resumen de lo que se ha realizado.

- 3. Introducción y motivación. ¿Qué problema se aborda y por qué es importante?

- 4. Descripción del problema y estado del arte. ¿Cómo se ha abordado en la literatura? ¿Qué métodos existen? Las referencias deben ser reales (artículos, libros, documentación).

- 5. Formulación matemática. Variables de decisión, función objetivo y restricciones. Tipo de problema: continuo o discreto, convexo o no, con o sin restricciones, determinista o estocástico. Esta es una de las partes más importantes.

- 6. Métodos. Descripción de cada método utilizado, pseudocódigo, parámetros, complejidad y justificación de su elección.

- 7. Implementación. Organización del código, decisiones de diseño y dificultades encontradas.

- 8. Datos. Origen, descripción, preprocesado y particiones utilizadas.

- 9. Experimentación y resultados. Diseño de los experimentos, métricas, tablas y gráficas (curvas de convergencia, comparativas, visualización de soluciones).

- 10. Discusión. Interpretación de los resultados: qué método funciona mejor, en qué condiciones y por qué. Limitaciones del trabajo.

- 11. Conclusiones y trabajo futuro.

- 12. Referencias y anexos.

## Buenas prácticas en la experimentación

- Compara siempre con una referencia. Una línea base sencilla (búsqueda aleatoria, heurística voraz, método de librería) permite saber si el método propuesto aporta algo.

- Repite los experimentos. Los métodos estocásticos (metaheurísticas, SGD) deben ejecutarse varias veces con distintas semillas. Informa de media, desviación típica y, cuando proceda, aplica tests estadísticos.

- Compara en igualdad de condiciones. Mismo presupuesto de evaluaciones de la función objetivo o de tiempo para todos los métodos.

- Separa los datos. Cuando haya aprendizaje, ajusta con entrenamiento y validación y reporta el resultado final en un conjunto de test que no se haya usado antes.

- Haz el trabajo reproducible. Fija semillas, documenta versiones de librerías y deja scripts que regeneren tablas y figuras.

- Analiza, no solo muestres. Cada tabla o gráfica debe ir acompañada de una interpretación.


## 4. Uso de la inteligencia artificial

Se recomienda utilizar herramientas de IA (asistentes conversacionales, asistentes de programación, agentes) para desarrollar el proyecto. Bien utilizada, la IA permite aprender más rápido y abordar proyectos más avanzados. Algunos usos recomendados:

- Comprender conceptos y métodos, pedir explicaciones alternativas y ejemplos.

- Hacer una primera exploración del estado del arte y localizar trabajos relevantes.

- Generar, depurar, documentar y optimizar código.

- Diseñar experimentos y revisar el análisis de resultados.

- Ayudar en la redacción y revisión de la memoria.

## Condiciones de uso:

- El autor es responsable de todo. Hay que entender todo el código y todo el contenido de la memoria. En la evaluación (y en la presentación, si se hace) se podrá preguntar por cualquier parte del proyecto.

- Verifica siempre. La IA comete errores y puede inventar referencias bibliográficas, datos o resultados. Todas las referencias deben existir y haberse consultado, y todos los resultados deben proceder de ejecuciones reales del código.

- Los métodos deben implementarse. Cuando el proyecto pide implementar un método, la implementación debe formar parte del proyecto (aunque se haya desarrollado con ayuda de la IA). Las librerías se usan para comparar y validar.

- Declara el uso. La memoria incluirá un apartado indicando qué herramientas se han usado, para qué tareas y con algunos ejemplos de las peticiones más relevantes, así como una valoración crítica de su utilidad y de los errores detectados.

## 5. Criterios de evaluación

Se valorarán, entre otros, los siguientes aspectos:

- Comprensión y formulación del problema: claridad y corrección del modelo de optimización.

- Investigación: calidad y actualidad de la revisión del estado del arte.

- Métodos e implementación: adecuación de los métodos, corrección y calidad del código, grado de dificultad abordado.

- Experimentación con datos reales: rigor del diseño experimental, comparativas y análisis estadístico.

- Análisis y conclusiones: interpretación crítica de los resultados y de las limitaciones.

- Calidad de la memoria: organización, claridad, figuras y referencias.

- Ambición y originalidad: aportaciones propias, extensiones y aplicación a problemas de nuestro entorno.

- Presentación (opcional): claridad de la exposición, demostración y respuesta a preguntas.



## 6. Propuesta Elegida

# A4. Optimización de redes neuronales con algoritmos genéticos (neuroevolución)

**Dificultad:** ★★★★☆ Alta

## Descripción
Los algoritmos evolutivos permiten optimizar redes neuronales sin necesidad de calcular gradientes: pueden optimizar los pesos, la arquitectura o los hiperparámetros. Son especialmente útiles cuando la función objetivo no es diferenciable, como ocurre en muchos problemas de control o de aprendizaje por refuerzo.

## Qué se pide
- Implementar un algoritmo genético para optimizar los pesos de una red pequeña y compararlo con la retropropagación.
- Implementar un algoritmo genético para optimizar la arquitectura y los hiperparámetros de una red mayor entrenada con gradiente.
- Diseñar operadores de cruce y mutación adecuados a la representación de una red neuronal.
- Analizar el coste computacional y aplicar paralelización.

## Métodos y técnicas
Algoritmos genéticos, estrategias evolutivas, CMA-ES, NEAT (*NeuroEvolution of Augmenting Topologies*).

## Datos reales sugeridos
- Conjuntos tabulares del *UCI Machine Learning Repository*.
- Entornos de control de *Gymnasium* (`CartPole`, `LunarLander`, `BipedalWalker`), donde no existe un gradiente directo.

## Para ir más allá
Implementar NEAT; comparar con las estrategias evolutivas usadas como alternativa escalable al aprendizaje por refuerzo.