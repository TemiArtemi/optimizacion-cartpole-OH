# PLAN DE TRABAJO

Fichero vivo y breve: refleja **qué está hecho** y **qué toca a continuación**. Se actualiza al cerrar cada subtarea. Los detalles técnicos no van aquí, sino en el chat y en la memoria.

**Última actualización:** 2026-10-27

---

## 1. Estado del proyecto

**Propuesta A4 — Neuroevolución:** algoritmos genéticos (AG) para optimizar redes neuronales en dos vertientes:

1. Pesos de una red pequeña en *CartPole-v1* (NumPy puro), comparados con retropropagación.
2. Arquitectura e hiperparámetros de una red mayor entrenada con gradiente (PyTorch) sobre datos tabulares del UCI Repository.

### Hecho

- [x] Repositorio inicial: `enunciado.md`, `AGENTS.md`, `requirements.txt`, `.gitignore`.
- [x] Fase 0 (2026-10-27): `red_base.py` migrado a `src/` (red, operadores y evaluación en módulos separados, en inglés, sin comentarios, con type hints), fitness multi-episodio con semilla y `render_mode` corregido. Verificado con `python -m tests.smoke_test`.

### Decisiones de diseño (provisionales, confirmar con el equipo)

- Baseline de retropropagación (Fase 2): **REINFORCE en NumPy puro** (policy gradient), no PyTorch.
- PyTorch se usa solo en la Fase 3 (entrenar con gradiente sobre UCI).
- Sin ampliaciones (NEAT, CMA-ES, visualización, más entornos) hasta tener la base hecha.

### Tarea inmediata

- **Fase 1:** crear `src/genetic/selection.py` (torneo y ruleta) y `src/genetic/engine.py` (población, elitismo, bucle evolutivo).

---

## 2. Plan por fases

### Fase 0 — Estructura e higiene del código ✅

- [x] Migrado a `src/networks/numpy_mlp.py`, `src/genetic/operators.py`, `src/environments/cartpole_eval.py`.
- [x] Código en inglés, cero comentarios `#`, docstrings de una línea, type hints.
- [x] Fitness multi-episodio con semilla; `render_mode` corregido en Gymnasium.
- [x] `requirements.txt` con versiones fijadas, `.gitignore` ampliado, README esqueleto.
- [x] Verificación: `python -m tests.smoke_test` en verde.

### Fase 1 — Motor genético desde cero

- [ ] `src/genetic/selection.py`: selección por torneo y por ruleta.
- [ ] `src/genetic/engine.py`: población, elitismo, condición de parada e historia de convergencia, desacoplado del dominio (solo recibe una función fitness).
- [ ] Línea base: búsqueda aleatoria con el mismo presupuesto de evaluaciones.
- [ ] Verificación: el AG supera claramente al azar en curvas de convergencia.

### Fase 2 — Experimentación rigurosa en CartPole

- [ ] Evaluación paralela (`multiprocessing`/`concurrent.futures`) y medición de *speedup*.
- [ ] Experimentos con múltiples semillas: media, desviación típica, curvas de convergencia.
- [ ] Comparación de operadores: torneo vs ruleta; cruce un punto vs uniforme vs BLX-α; mutación gaussiana.
- [ ] Baseline REINFORCE en NumPy frente al AG (misma red, mismo presupuesto de episodios).
- [ ] Scripts en `experiments/`, figuras en `results/figures/`, logs en `results/logs/`.

### Fase 3 — Vertiente UCI con gradiente

- [ ] Elegir dataset tabular del UCI y documentar origen y licencia.
- [ ] `src/networks/torch_mlp.py` entrenado con PyTorch.
- [ ] AG sobre arquitectura e hiperparámetros (cromosoma mixto discreto/continuo) como caja negra.
- [ ] Comparar con búsqueda aleatoria y con una configuración por defecto.

### Fase 4 — Memoria y entregables

- [ ] Memoria según la estructura del enunciado (incluye apartado de uso de IA).
- [ ] README definitivo con instalación y ejecución.
- [ ] Regeneración de tablas y figuras a partir de scripts reproducibles.
- [ ] Presentación opcional (decidir cuando haya resultados).

### Ampliaciones posibles (sin comprometer)

- NEAT, CMA-ES, animación/visualización del agente, LunarLander. Decidir cuando la base esté lista.

---

## 3. Reglas de actualización

1. Al cerrar cada subtarea: pasarla a **Hecho**, actualizar la **Tarea inmediata** y la fecha; los cambios de plan se acuerdan en el chat antes de escribirlos aquí.
2. Mantener el fichero por debajo de ~100 líneas: si crece, resumir la fase cerrada a una o dos líneas o trasladar los detalles a la memoria.
3. Este fichero registra **estado y próximos pasos**, no explicaciones técnicas ni resultados (eso va en el chat o en la memoria).
