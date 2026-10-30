# PLAN DE TRABAJO

Fichero vivo y breve: refleja **qué está hecho** y **qué toca a continuación**. Se actualiza al cerrar cada subtarea. Los detalles técnicos no van aquí, sino en el chat y en la memoria.

**Última actualización:** 2026-10-30

---

## 1. Estado del proyecto

**Propuesta A4 — Neuroevolución:** algoritmos genéticos (AG) para optimizar redes neuronales en dos vertientes:

1. Pesos de una red pequeña en *CartPole-v1* (NumPy puro), comparados con retropropagación.
2. Arquitectura e hiperparámetros de una red mayor entrenada con gradiente (PyTorch) sobre datos tabulares del UCI Repository.

### Hecho

- [x] Repositorio inicial: `enunciado.md`, `AGENTS.md`, `requirements.txt`, `.gitignore`.
- [x] Fase 0 (2026-10-27): `red_base.py` migrado a `src/` (red, operadores y evaluación en módulos separados, en inglés, sin comentarios, con type hints), fitness multi-episodio con semilla y `render_mode` corregido. Verificado con `python -m tests.smoke_test`.
- [x] Fase 1 (2026-10-30): motor genético desde cero (selección, elitismo, presupuesto de evaluaciones), línea base de búsqueda aleatoria, 9 tests y verificación en CartPole.

### Decisiones de diseño (provisionales, confirmar con el equipo)

- Baseline de retropropagación (Fase 2): **REINFORCE en NumPy puro** (policy gradient), no PyTorch.
- PyTorch se usa solo en la Fase 3 (entrenar con gradiente sobre UCI).
- Sin ampliaciones (NEAT, CMA-ES, visualización, más entornos) hasta tener la base hecha.

### Tarea inmediata

- **Fase 2:** comparación de operadores: implementar cruce uniforme y BLX-α en `src/genetic/operators.py`, y el experimento `experiments/operator_comparison.py` (torneo vs ruleta; un punto vs uniforme vs BLX-α).

---

## 2. Plan por fases

### Fase 0 — Estructura e higiene del código ✅

- [x] Migrado a `src/networks/numpy_mlp.py`, `src/genetic/operators.py`, `src/environments/cartpole_eval.py`.
- [x] Código en inglés, cero comentarios `#`, docstrings de una línea, type hints.
- [x] Fitness multi-episodio con semilla; `render_mode` corregido en Gymnasium.
- [x] `requirements.txt` con versiones fijadas, `.gitignore` ampliado, README esqueleto.
- [x] Verificación: `python -m tests.smoke_test` en verde.

### Fase 1 — Motor genético desde cero ✅

- [x] `src/genetic/selection.py` (torneo y ruleta), `src/genetic/engine.py` (elitismo, presupuesto de evaluaciones, historia de convergencia) y `src/genetic/random_search.py`.
- [x] 9 tests (`python3 -m tests.phase1_test`) y verificación en CartPole (`experiments/phase1_check.py`, figura en `results/figures/`).
- Decisión: la verificación usa una red 4-16-16-2 y fitness robusto (condiciones iniciales nuevas en cada evaluación, sembradas por ejecución); con la red pequeña (4, 8, 2) la búsqueda aleatoria satura el entorno y la comparación no es informativa.

### Fase 2 — Experimentación rigurosa en CartPole

- [x] Evaluación paralela (`concurrent.futures`) con semillas preasignadas y *speedup* medido en `experiments/parallel_speedup.py` (figura y log en `results/`).
- [x] Experimentos con múltiples semillas: media, desviación típica, curvas de convergencia (utilidades en `src/utils/` y validación en `experiments/phase1_check.py`).
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
