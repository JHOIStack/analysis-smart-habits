# analysis-smart-habits

## Descripción
API de Machine Learning para análisis de hábitos inteligentes. Permite realizar clasificación supervisada y clustering no supervisado de usuarios, integrando modelos de ML con FastAPI.

## Endpoints principales

- `/ml/supervised`: Retorna la clasificación de perfiles de usuario usando un modelo Random Forest.
- `/ml/unsupervised`: Retorna los clusters de usuarios generados por KMeans según edad, hábitos e interacciones.

## Ejecución local

Para iniciar la API ejecuta:

```bash
uvicorn main:app --reload
```

La API estará disponible en `http://localhost:8000`.

## Integración y CORS
La API permite peticiones desde cualquier origen (CORS habilitado) para facilitar la integración con frontends como Express o visualizaciones locales.

## Visualización
Los resultados pueden visualizarse usando los scripts en la carpeta `visualizations/` o el archivo `mexico_users_map.html`.

## Estructura de carpetas

- `main.py`: API principal FastAPI
- `main_supervised.py`: Clasificación supervisada
- `main_unsupervised.py`: Clustering no supervisado
- `visualizations/`: Scripts para mapas y gráficos
- `db/`: Modelos y conexión a base de datos
- `utils/`: Utilidades para manejo de datos

## Requisitos
Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Autor
JHOIStack
