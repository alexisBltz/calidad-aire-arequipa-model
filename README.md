# Calidad del Aire Arequipa - API y Análisis

Este proyecto implementa una API REST para predecir la calidad del aire en Arequipa usando modelos de machine learning entrenados con datos locales. Incluye notebooks para análisis exploratorio y scripts para predicción.

---

## Despliegue con Docker

1. Instala Docker y Docker Compose.
2. En la raíz del proyecto, ejecuta:

    ```powershell
    docker-compose up --build
    ```

3. Accede a la API en: [http://localhost:8000](http://localhost:8000)
4. La documentación interactiva estará disponible en: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Ejecución local (sin Docker)

1. Instala Python 3.9+ y pip.
2. Instala las dependencias:

    ```powershell
    pip install -r requirements.txt
    ```

3. Ejecuta la API:

    ```powershell
    cd api
    python app.py
    ```

4. Accede a la API en: [http://localhost:8000](http://localhost:8000)

---

## Endpoints principales

- **GET /**  
  Información básica y enlaces de la API.

- **GET /health**  
  Estado de salud del sistema y del predictor.

- **GET /info**  
  Información sobre los modelos cargados y características usadas.

- **POST /predict**  
  Realiza una predicción de calidad del aire a partir de datos meteorológicos.

### Ejemplo de request

```json
{
  "temperatura": 20.5,
  "humedad": 60,
  "presion": 1012,
  "velocidad_viento": 2.5,
  "direccion_viento": 180,
  "precipitacion": 0.0,
  "fecha": "2025-07-09 14:00:00"
}
```

### Ejemplo de respuesta

```json
{
  "pm10": 45.2,
  "pm25": 22.1,
  "timestamp": "2025-07-09T14:00:00",
  "ubicacion": "Socabaya, Arequipa",
  "estado": "Bueno"
}
```

- **GET /predict/example**  
  Devuelve una predicción de ejemplo con datos típicos de Arequipa.

---

## Breve explicación del endpoint `/predict`

El endpoint `/predict` recibe un JSON con variables meteorológicas (temperatura, humedad, presión, velocidad y dirección del viento, precipitación y fecha opcional) y retorna la predicción de concentración de PM10 y PM2.5, junto con el estado de calidad del aire según estándares peruanos.

- **Variables requeridas:**
  - `temperatura` (°C)
  - `humedad` (%)
  - `presion` (hPa)
- **Variables opcionales:**
  - `velocidad_viento` (km/h, por defecto 3.0)
  - `direccion_viento` (grados, por defecto 180)
  - `precipitacion` (mm, por defecto 0.0)
  - `fecha` (formato ISO o 'YYYY-MM-DD HH:MM:SS')

La respuesta incluye los valores predichos de PM10 y PM2.5, el estado de calidad del aire y la ubicación.

---

## Notebooks

En la carpeta `notebooks/` encontrarás análisis exploratorio, preparación de datos y ejemplos de uso de los modelos.

---

Para dudas o mejoras, abre un issue o contacta al autor.
