#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API REST - Sistema de Predicción de Calidad del Aire Arequipa
============================================================

API FastAPI para predicción de calidad del aire en tiempo real.
"""

import sys
import os
from datetime import datetime
from typing import Dict, Optional

# Configurar el path para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Importar el predictor
try:
    from src.predictor import PredictorCalidadAire
except ImportError as e:
    print(f"Error importing predictor: {e}")
    sys.exit(1)

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de Calidad del Aire - Arequipa",
    description="API REST para predicción de calidad del aire en Arequipa usando modelos de machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Inicializar el predictor globalmente
predictor = None

@app.on_event("startup")
async def startup_event():
    """Inicializar el predictor al arrancar la API."""
    global predictor
    try:
        predictor = PredictorCalidadAire()
        print("✅ Predictor cargado exitosamente")
    except Exception as e:
        print(f"❌ Error cargando predictor: {e}")
        # No salir del programa, continuar sin predictor

# Modelos Pydantic para la API
class DatosMeteorologicos(BaseModel):
    temperatura: float
    humedad: float  
    presion: float
    velocidad_viento: Optional[float] = 3.0
    direccion_viento: Optional[float] = 180.0
    precipitacion: Optional[float] = 0.0
    fecha: Optional[str] = None  # Nuevo campo opcional para la fecha/hora

class RespuestaPrediccion(BaseModel):
    pm10: float
    pm25: float
    timestamp: str
    ubicacion: str
    estado: str

@app.get("/")
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "mensaje": "API de Calidad del Aire - Arequipa",
        "version": "1.0.0",
        "documentacion": "/docs",
        "endpoints": ["/predict", "/health", "/info"]
    }

@app.get("/health")
async def health_check():
    """Health check del sistema."""
    global predictor
    
    status = "healthy" if predictor and predictor.cargado else "unhealthy"
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "predictor_loaded": predictor is not None and predictor.cargado if predictor else False
    }

@app.get("/info")
async def get_info():
    """Información del sistema y modelos."""
    global predictor
    
    if not predictor or not predictor.cargado:
        return {"error": "Predictor no disponible"}
    
    return {
        "sistema": "Predicción de Calidad del Aire - Arequipa",
        "modelos": {
            contaminante: {
                "tipo": info["tipo"],
                "metricas": info["metricas"]
            }
            for contaminante, info in predictor.modelos.items()
        },
        "caracteristicas": len(predictor.caracteristicas),
        "ubicacion": "Socabaya, Arequipa"
    }

@app.post("/predict", response_model=RespuestaPrediccion)
async def predecir_calidad_aire(datos: DatosMeteorologicos):
    """
    Predecir calidad del aire basado en datos meteorológicos.
    
    Parámetros:
    - temperatura: Temperatura en °C
    - humedad: Humedad relativa en %
    - presion: Presión barométrica en hPa
    - velocidad_viento: Velocidad del viento en km/h (opcional)
    - direccion_viento: Dirección del viento en grados (opcional)
    - precipitacion: Precipitación en mm (opcional)
    - fecha: Fecha y hora de la medición (opcional, formato ISO o 'YYYY-MM-DD HH:MM:SS')
    """
    global predictor
    
    if not predictor or not predictor.cargado:
        raise HTTPException(status_code=503, detail="Predictor no disponible")
    
    try:
        # Convertir datos a diccionario
        datos_dict = {
            "temperatura": datos.temperatura,
            "humedad": datos.humedad,
            "presion": datos.presion,
            "velocidad_viento": datos.velocidad_viento,
            "direccion_viento": datos.direccion_viento,
            "precipitacion": datos.precipitacion
        }
        if datos.fecha:
            datos_dict["fecha"] = datos.fecha  # Pasar la fecha si se envía
        
        # Realizar predicción
        resultado = predictor.predecir(datos_dict)
        
        # Extraer valores
        pm10_valor = resultado["predicciones"]["PM10"]["valor"]
        pm25_valor = resultado["predicciones"]["PM2.5"]["valor"]
        pm10_nivel = resultado["predicciones"]["PM10"]["clasificacion"]
        
        return RespuestaPrediccion(
            pm10=pm10_valor,
            pm25=pm25_valor,
            timestamp=resultado["timestamp"],
            ubicacion=resultado["ubicacion"],
            estado=pm10_nivel
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/predict/example")
async def ejemplo_prediccion():
    """Ejemplo de predicción con datos típicos de Arequipa."""
    datos_ejemplo = DatosMeteorologicos(
        temperatura=18.5,
        humedad=45.0,
        presion=1013.2,
        velocidad_viento=3.2,
        direccion_viento=225,
        precipitacion=0.0
    )
    
    return await predecir_calidad_aire(datos_ejemplo)

if __name__ == "__main__":
    print("🚀 Iniciando API de Calidad del Aire - Arequipa")
    print("📍 Documentación disponible en: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
