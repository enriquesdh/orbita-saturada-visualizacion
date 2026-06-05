# Órbita saturada

Proyecto de visualización de datos sobre el crecimiento, usos y sostenibilidad del espacio cercano a la Tierra.

## Estructura

- `data/raw`: datos originales.
- `data/interim`: datos limpiados intermedios.
- `data/processed`: datos enriquecidos.
- `data/flourish`: CSV finales utilizados en Flourish.
- `notebooks`: análisis exploratorio inicial.
- `scripts`: scripts de descarga y generación de datasets.
- `requirements.txt`: dependencias principales.

## Fuentes de datos

- CelesTrak SATCAT.
- UCS Satellite Database.
- Our World in Data / UNOOSA.

## Reproducción

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/00_download_data.py
python scripts/01_build_flourish_datasets.py