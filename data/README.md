# Dataset Letter Recognition (UCI)

## Origen

- **Nombre:** Letter Recognition
- **Repositorio:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/59/letter+recognition)
- **DOI:** [10.24432/C5ZP40](https://doi.org/10.24432/C5ZP40)
- **Autor:** David Slate (1991)
- **Referencia:** P. W. Frey and D. J. Slate, "Letter Recognition Using Holland-style Adaptive Classifiers", *Machine Learning*, 6(2), 1991.
- **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Descripción

20 000 instancias de imágenes de las 26 letras mayúsculas del alfabeto (A–Z), representadas por 16 atributos numéricos enteros (momentos estadísticos y conteos de bordes, escalados al rango 0–15). Es una tarea de clasificación **multiclase** con 26 clases balanceadas (~750 muestras por clase) y **sin valores missing**.

## Preprocesado

Generar las particiones (desde la raíz del repositorio):

```bash
python -m data.prepare_letter
```

El script realiza:

1. **Descarga** del archivo oficial `letter-recognition.data` desde UCI (cacheado en `data/raw/`).
2. **Codificación** de la letra (A–Z) como índice entero 0–25.
3. **Partición:** las primeras 16 000 filas (bloque de entrenamiento oficial) se barajan con la semilla `0` → 12 000 train / 4 000 val; las últimas 4 000 filas (test oficial) quedan intactas.
4. **Estandarización:** cada columna se centra y escala con la media y la desviación calculadas **solo** sobre el conjunto de train; esos mismos parámetros se aplican a val y test.

## Archivos generados

- `data/letter/train.npz`, `data/letter/val.npz`, `data/letter/test.npz` — claves `features` (float64) y `labels` (int64).

No se versionan (ver `.gitignore`); se regeneran en cualquier momento con el script.
