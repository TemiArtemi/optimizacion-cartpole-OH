# Optimización de redes neuronales con algoritmos genéticos (neuroevolución)

Proyecto de la asignatura **Optimización y Heurística** (Grado en Ciencia e Ingeniería de Datos, ULPGC) — propuesta A4.

El proyecto desarrolla desde cero algoritmos genéticos para optimizar redes neuronales en dos vertientes:

1. Optimización directa de los pesos de una red pequeña en el entorno *CartPole-v1* de Gymnasium (función objetivo no diferenciable), comparada con retropropagación.
2. Optimización de la arquitectura e hiperparámetros de una red mayor entrenada con gradiente sobre un conjunto de datos tabular del UCI Machine Learning Repository.

## Requisitos

- Python 3.11 o superior.
- Dependencias listadas en `requirements.txt`.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Estructura del repositorio

```
├── data/                    datos y scripts de descarga/preprocesado
├── results/                 figures/ y logs/ de los experimentos
├── src/
│   ├── networks/            redes neuronales (NumPy, PyTorch)
│   ├── genetic/             operadores, selección y motor evolutivo
│   ├── environments/        evaluación en CartPole
│   └── utils/               métricas y gráficas
├── tests/                   comprobaciones rápidas
├── experiments/             scripts de experimentos
├── PLAN.md                  estado del proyecto y plan vivo
├── AGENTS.md                guía de operación y reglas de desarrollo
└── enunciado.md             enunciado de la asignatura
```

## Ejecución

Comprobación rápida de la implementación (desde la raíz del repositorio):

```bash
python -m tests.smoke_test
```

## Plan de trabajo

El estado actual del proyecto y las tareas pendientes se mantienen en [`PLAN.md`](PLAN.md).

## Uso de IA

Apartado pendiente (obligatorio en la memoria final): descripción de las herramientas de IA utilizadas, tareas realizadas, ejemplos de peticiones relevantes y valoración crítica.
