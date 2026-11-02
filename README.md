# Optimización de redes neuronales con algoritmos genéticos (neuroevolución)

Proyecto de la asignatura **Optimización y Heurística** (Grado en Ciencia e Ingeniería de Datos, ULPGC) — propuesta A4.

El proyecto desarrolla desde cero algoritmos genéticos para optimizar redes neuronales en dos vertientes:

1. **Pesos en CartPole:** optimización directa de los pesos de una red pequeña en el entorno *CartPole-v1* de Gymnasium (función objetivo no diferenciable), comparada con búsqueda aleatoria y con el método de gradiente REINFORCE.
2. **Arquitectura e hiperparámetros en UCI:** optimización de la arquitectura y los hiperparámetros de una red mayor entrenada con descenso por gradiente (PyTorch) sobre el conjunto tabular *Letter Recognition* del UCI Machine Learning Repository, comparada con búsqueda aleatoria y con una configuración por defecto.

## Requisitos

- Python 3.11–3.13 (las versiones exactas de `requirements.txt` están fijadas para estos intérpretes; en el desarrollo se comprobó también con Python 3.14 usando `numpy==2.5.4` y `matplotlib==3.11.2` en lugar de los *pines* de `requirements.txt`).
- Dependencias listadas en `requirements.txt` (NumPy, Gymnasium, pygame, matplotlib, PyTorch).
- CPU multicor: la evaluación paralela de poblaciones usa `concurrent.futures.ProcessPoolExecutor`.

## Instalación

Linux / macOS / WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Preparación de los datos

Los datos de UCI no se versionan (tamaño y licencia): se descargan y preprocesan con un único comando, que genera `data/letter/{train,val,test}.npz` a partir del archivo oficial con la partición oficial y la estandarización ajustada solo sobre entrenamiento (ver `data/README.md` para origen, DOI y licencia):

```bash
python -m data.prepare_letter
```

## Estructura del repositorio

```
├── data/                    descarga y preprocesado del dataset UCI Letter Recognition
├── results/
│   ├── figures/             figuras de convergencia y speedup (en español)
│   └── logs/                registros en bruto de cada experimento
├── src/
│   ├── networks/            redes neuronales: numpy_mlp.py (CartPole), torch_mlp.py (UCI)
│   ├── genetic/             motor evolutivo, selección, operadores y cromosoma mixto
│   ├── baselines/           REINFORCE en NumPy puro (línea base de gradiente)
│   ├── environments/        evaluación en CartPole (serial y paralela)
│   ├── evaluation/          fitness de configuraciones UCI (entrenamiento en paralelo)
│   ├── uci/                 carga y preprocesado del dataset
│   └── utils/               métricas, gráficas y utilidades de semillas
├── tests/                   seis baterías de comprobación
├── experiments/             scripts de experimentos reproducibles
├── PLAN.md                  estado del proyecto y plan vivo
├── AGENTS.md                guía de operación y reglas de desarrollo
└── enunciado.md             enunciado de la asignatura
```

## Ejecución

### Comprobaciones (desde la raíz del repositorio)

```bash
python -m tests.smoke_test        # red NumPy, operadores y evaluación CartPole
python -m tests.phase1_test       # selección, motor, baseline aleatoria, cruces
python -m tests.parallel_test     # evaluación paralela y semillas preasignadas
python -m tests.utils_test        # métricas compartidas
python -m tests.reinforce_test    # REINFORCE (incluye verificación por diferencias finitas)
python -m tests.phase3_test       # dataset, torch_mlp, cromosoma mixto y evaluador UCI
```

Los tests de Fase 3 que dependen del dataset omiten con un mensaje explícito si `data/letter/` no existe; genere los datos antes de ejecutarlos.

### Experimentos

| Comando | Contenido | Duración aproximada (8 workers) |
| --- | --- | --- |
| `python -m experiments.parallel_speedup` | speedup de la evaluación paralela | ~3 min |
| `python -m experiments.phase1_check` | AG vs búsqueda aleatoria en CartPole (5 semillas) | ~7 min |
| `python -m experiments.operator_comparison` | torneo vs ruleta y cruces (4 configuraciones × 5 semillas) | ~10 min |
| `python -m experiments.reinforce_comparison` | AG vs aleatoria vs REINFORCE a igual presupuesto de episodios | ~8 min |
| `python -m experiments.phase3_check` | AG vs aleatoria vs configuración por defecto en Letter Recognition (5 semillas) | ~1 h 45 min |

Cada experimento guarda su figura en `results/figures/` y sus datos en bruto en `results/logs/`, y es determinista por semillas: re-ejecutarlo reproduce exactamente los mismos valores (excepto `parallel_speedup`, cuyo log contiene tiempos de pared dependientes de la máquina).

## Plan de trabajo

El estado actual del proyecto y las tareas pendientes se mantienen en [`PLAN.md`](PLAN.md).

## Uso de IA

Apartado obligatorio en la memoria final: descripción de las herramientas de IA utilizadas, tareas realizadas, ejemplos de peticiones relevantes y valoración crítica.
