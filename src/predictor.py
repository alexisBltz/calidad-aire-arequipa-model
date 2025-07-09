#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictor de Calidad del Aire - Arequipa
========================================

Módulo principal para predicción de calidad del aire en Arequipa.
Incluye modelos de machine learning y clasificación según estándares peruanos.

Autor: Sistema de Monitoreo Ambiental
Fecha: Julio 2025
Región: Arequipa, Perú
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union

# Agregar el directorio actual al path para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

class PredictorCalidadAire:
    """
    Predictor principal de calidad del aire para Arequipa.
    
    Características:
    - Predicción de PM10 y PM2.5 usando modelos entrenados
    - Clasificación según estándares peruanos (ECA)
    - Alertas automáticas por niveles de contaminación
    - Historial de predicciones
    - Validación de datos de entrada
    """
    
    def __init__(self, models_path: Optional[str] = None):
        """
        Inicializar el predictor.
        
        Args:
            models_path (str): Ruta a los modelos entrenados
        """
        self.models_path = models_path or os.path.join(current_dir, '..', 'models')
        self.modelos = {}
        self.caracteristicas = []
        self.historial_predicciones = []
        self.cargado = False
        
        # Estándares de Calidad Ambiental (ECA) del Perú
        self.umbrales_eca = {
            'PM10': [
                (0, 50, 'Bueno', 'Verde', '🟢'),
                (51, 100, 'Moderado', 'Amarillo', '🟡'),
                (101, 167, 'Dañino para Grupos Sensibles', 'Naranja', '🟠'),
                (168, 250, 'Dañino', 'Rojo', '🔴'),
                (251, float('inf'), 'Muy Dañino', 'Morado', '🟣')
            ],
            'PM2.5': [
                (0, 25, 'Bueno', 'Verde', '🟢'),
                (26, 50, 'Moderado', 'Amarillo', '🟡'),
                (51, 75, 'Dañino para Grupos Sensibles', 'Naranja', '🟠'),
                (76, 100, 'Dañino', 'Rojo', '🔴'),
                (101, float('inf'), 'Muy Dañino', 'Morado', '🟣')
            ]
        }
        
        # Cargar modelos automáticamente
        self.cargar_modelos()
    
    def cargar_modelos(self) -> bool:
        """
        Cargar modelos entrenados y características.
        
        Returns:
            bool: True si se cargaron correctamente, False en caso contrario
        """
        try:
            # Rutas de archivos
            modelo_pm10_path = os.path.join(self.models_path, 'modelo_pm10_arequipa_maximizado.pkl')
            modelo_pm25_path = os.path.join(self.models_path, 'modelo_pm25_arequipa_maximizado.pkl')
            caracteristicas_path = os.path.join(self.models_path, 'caracteristicas_modelo_arequipa.json')
            
            # Verificar que existen los archivos
            archivos_requeridos = [modelo_pm10_path, modelo_pm25_path, caracteristicas_path]
            for archivo in archivos_requeridos:
                if not os.path.exists(archivo):
                    print(f"❌ Archivo no encontrado: {archivo}")
                    return False
            
            # Cargar modelos
            self.modelos['PM10'] = {
                'modelo': joblib.load(modelo_pm10_path),
                'tipo': 'RandomForest',
                'metricas': {'R2': 0.35, 'RMSE': 15.2, 'MAE': 12.1}
            }
            
            self.modelos['PM2.5'] = {
                'modelo': joblib.load(modelo_pm25_path),
                'tipo': 'RandomForest', 
                'metricas': {'R2': 0.28, 'RMSE': 8.7, 'MAE': 6.9}
            }
            
            # Cargar características
            with open(caracteristicas_path, 'r', encoding='utf-8') as f:
                self.caracteristicas = json.load(f)
            
            self.cargado = True
            print(f"✅ Modelos cargados exitosamente:")
            print(f"   • PM10: {self.modelos['PM10']['tipo']} (R² = {self.modelos['PM10']['metricas']['R2']})")
            print(f"   • PM2.5: {self.modelos['PM2.5']['tipo']} (R² = {self.modelos['PM2.5']['metricas']['R2']})")
            print(f"   • Características: {len(self.caracteristicas)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            self.cargado = False
            return False
    
    def predecir(self, datos: Dict[str, Union[float, str]]) -> Dict:
        """
        Realizar predicción de calidad del aire.
        
        Args:
            datos (dict): Diccionario con datos meteorológicos
                         Ejemplo: {
                             'FECHA_INICIO': '2021-12-15 14:00:00',
                             'TEMPERATURA': 18.5,
                             'HUMEDAD_RELATIVA': 45.0,
                             'PRESION_BAROMETRICA': 745.0
                         }
        
        Returns:
            dict: Resultado de la predicción con valores y clasificaciones
        """
        if not self.cargado:
            return {
                'error': 'Modelos no cargados',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Validar y procesar datos de entrada
            datos_procesados = self._procesar_datos_entrada(datos)
            
            # Crear DataFrame con las características esperadas
            df_entrada = pd.DataFrame([datos_procesados])
            
            # Asegurar que tiene todas las características necesarias
            for caracteristica in self.caracteristicas:
                if caracteristica not in df_entrada.columns:
                    df_entrada[caracteristica] = 0.0
            
            # Reordenar columnas según el orden de entrenamiento
            df_entrada = df_entrada[self.caracteristicas]
            
            # Realizar predicciones
            resultados = {
                'timestamp': datetime.now().isoformat(),
                'ubicacion': 'Socabaya, Arequipa',
                'datos_entrada': datos,
                'predicciones': {},
                'alertas': []
            }
            
            for contaminante in ['PM10', 'PM2.5']:
                try:
                    modelo = self.modelos[contaminante]['modelo']
                    valor_predicho = modelo.predict(df_entrada)[0]
                    valor_predicho = max(0, valor_predicho)  # No permitir valores negativos
                    
                    # Clasificar según estándares ECA
                    clasificacion = self._clasificar_calidad_aire(contaminante, valor_predicho)
                    
                    resultados['predicciones'][contaminante] = {
                        'valor': round(valor_predicho, 1),
                        'unidad': 'μg/m³',
                        'clasificacion': clasificacion['nivel'],
                        'color': clasificacion['color'],
                        'emoji': clasificacion['emoji'],
                        'descripcion': clasificacion.get('descripcion', ''),
                        'modelo_usado': self.modelos[contaminante]['tipo'],
                        'confianza': self.modelos[contaminante]['metricas']['R2']
                    }
                    
                    # Generar alertas si es necesario
                    if clasificacion['requiere_alerta']:
                        resultados['alertas'].append({
                            'contaminante': contaminante,
                            'nivel': clasificacion['nivel'],
                            'valor': valor_predicho,
                            'limite': clasificacion.get('limite_superior', 0),
                            'mensaje': f"Nivel {clasificacion['nivel']} de {contaminante} detectado ({valor_predicho:.1f} μg/m³)"
                        })
                
                except Exception as e:
                    resultados['predicciones'][contaminante] = {
                        'error': str(e),
                        'valor': None
                    }
            
            # Guardar en historial
            self.historial_predicciones.append(resultados)
            
            return resultados
            
        except Exception as e:
            return {
                'error': f'Error en predicción: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _procesar_datos_entrada(self, datos: Dict) -> Dict:
        """
        Procesar y validar datos de entrada.
        
        Args:
            datos (dict): Datos crudos de entrada
            
        Returns:
            dict: Datos procesados y validados
        """
        import numpy as np
        
        datos_procesados = {}
        
        # Obtener temperatura, humedad y presión de los datos de entrada
        temperatura = float(datos.get('temperatura', datos.get('TEMPERATURA', 18.0)))
        humedad = float(datos.get('humedad', datos.get('HUMEDAD_RELATIVA', 50.0)))
        presion = float(datos.get('presion', datos.get('PRESION_BAROMETRICA', 745.0)))
        
        # Validar rangos meteorológicos
        datos_procesados['TEMPERATURA'] = self._validar_rango_meteorologico('TEMPERATURA', temperatura)
        datos_procesados['HUMEDAD_RELATIVA'] = self._validar_rango_meteorologico('HUMEDAD_RELATIVA', humedad)
        datos_procesados['PRESION_BAROMETRICA'] = self._validar_rango_meteorologico('PRESION_BAROMETRICA', presion)
        
        # Procesar fecha/hora
        fecha = datetime.now()  # Valor por defecto
        
        if 'fecha' in datos or 'FECHA_INICIO' in datos:
            try:
                fecha_str = datos.get('fecha', datos.get('FECHA_INICIO'))
                if isinstance(fecha_str, str):
                    fecha = pd.to_datetime(fecha_str)
                elif fecha_str is not None:
                    fecha = fecha_str
            except:
                fecha = datetime.now()
        
        # Extraer características temporales
        hora = fecha.hour
        dia = fecha.day
        mes = fecha.month
        dia_semana = fecha.weekday()
        
        # Características temporales básicas
        datos_procesados['hora'] = hora
        datos_procesados['día'] = dia
        datos_procesados['mes'] = mes
        datos_procesados['día_semana'] = dia_semana
        datos_procesados['es_fin_semana'] = 1 if dia_semana >= 5 else 0
        
        # Transformaciones trigonométricas (esto es clave para que funcione bien)
        datos_procesados['hora_sin'] = np.sin(2 * np.pi * hora / 24)
        datos_procesados['hora_cos'] = np.cos(2 * np.pi * hora / 24)
        datos_procesados['mes_sin'] = np.sin(2 * np.pi * mes / 12)
        datos_procesados['mes_cos'] = np.cos(2 * np.pi * mes / 12)
        
        return datos_procesados
    
    def _validar_rango_meteorologico(self, variable: str, valor: float) -> float:
        """
        Validar que los valores meteorológicos estén en rangos realistas.
        
        Args:
            variable (str): Nombre de la variable
            valor (float): Valor a validar
            
        Returns:
            float: Valor validado
        """
        rangos = {
            'TEMPERATURA': (-10, 45),     # °C para Arequipa
            'HUMEDAD_RELATIVA': (0, 100), # %
            'PRESION_BAROMETRICA': (700, 800)  # hPa para altitud de Arequipa
        }
        
        if variable in rangos:
            min_val, max_val = rangos[variable]
            return max(min_val, min(max_val, float(valor)))
        
        return float(valor)
    
    def _clasificar_calidad_aire(self, contaminante: str, valor: float) -> Dict:
        """
        Clasificar calidad del aire según estándares peruanos ECA.
        
        Args:
            contaminante (str): Tipo de contaminante
            valor (float): Valor de concentración
            
        Returns:
            dict: Información de la clasificación
        """
        if contaminante in self.umbrales_eca:
            for rango in self.umbrales_eca[contaminante]:
                limite_inf, limite_sup, nivel, color, emoji = rango
                
                if limite_inf <= valor <= limite_sup:
                    return {
                        'nivel': nivel,
                        'color': color,
                        'emoji': emoji,
                        'limite_inferior': limite_inf,
                        'limite_superior': limite_sup,
                        'valor': valor,
                        'requiere_alerta': nivel not in ['Bueno', 'Moderado'],
                        'descripcion': self._get_descripcion_nivel(nivel)
                    }
        
        return {
            'nivel': 'Desconocido',
            'color': 'Gris',
            'emoji': '⚪',
            'valor': valor,
            'requiere_alerta': True,
            'descripcion': 'Nivel no clasificado'
        }
    
    def _get_descripcion_nivel(self, nivel: str) -> str:
        """Obtener descripción detallada del nivel de calidad."""
        descripciones = {
            'Bueno': 'Calidad del aire satisfactoria. Riesgo mínimo para la salud.',
            'Moderado': 'Calidad del aire aceptable. Algunas personas sensibles pueden experimentar síntomas menores.',
            'Dañino para Grupos Sensibles': 'Personas sensibles pueden experimentar efectos en la salud.',
            'Dañino': 'Toda la población puede experimentar efectos en la salud.',
            'Muy Dañino': 'Emergencia sanitaria. Toda la población está en riesgo.'
        }
        return descripciones.get(nivel, 'Sin descripción disponible')
    
    def obtener_tendencia(self, horas_atras: int = 24) -> Dict:
        """
        Obtener tendencia de predicciones recientes.
        
        Args:
            horas_atras (int): Número de horas hacia atrás para analizar
            
        Returns:
            dict: Análisis de tendencias
        """
        if len(self.historial_predicciones) < 2:
            return {"error": "Datos insuficientes para calcular tendencia"}
        
        predicciones_recientes = self.historial_predicciones[-horas_atras:]
        tendencias = {}
        
        for contaminante in ['PM10', 'PM2.5']:
            valores = []
            for pred in predicciones_recientes:
                if (contaminante in pred.get('predicciones', {}) and 
                    pred['predicciones'][contaminante].get('valor') is not None):
                    valores.append(pred['predicciones'][contaminante]['valor'])
            
            if len(valores) >= 2:
                tendencia_valor = valores[-1] - valores[0]
                if abs(tendencia_valor) < 1:
                    tendencias[contaminante] = "Estable"
                elif tendencia_valor > 0:
                    tendencias[contaminante] = f"Aumentando (+{tendencia_valor:.1f})"
                else:
                    tendencias[contaminante] = f"Disminuyendo ({tendencia_valor:.1f})"
            else:
                tendencias[contaminante] = "Datos insuficientes"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'periodo_analizado': f"Últimas {len(predicciones_recientes)} predicciones",
            'tendencias': tendencias
        }
    
    def generar_reporte(self) -> Dict:
        """
        Generar reporte del estado del sistema.
        
        Returns:
            dict: Reporte completo del sistema
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'sistema': {
                'version': '1.0.0',
                'estado': 'Activo' if self.cargado else 'Inactivo',
                'modelos_cargados': len(self.modelos),
                'caracteristicas_modelo': len(self.caracteristicas)
            },
            'modelos': {
                contaminante: {
                    'tipo': info['tipo'],
                    'metricas': info['metricas'],
                    'estado': 'Operativo'
                }
                for contaminante, info in self.modelos.items()
            },
            'estadisticas': {
                'predicciones_realizadas': len(self.historial_predicciones),
                'ubicacion': 'Socabaya, Arequipa',
                'periodo_datos_entrenamiento': 'Diciembre 2021',
                'contaminantes_monitoreados': list(self.modelos.keys())
            },
            'salud_sistema': 'OK' if self.cargado else 'ERROR'
        }


# Clase simplificada para ejemplos rápidos
class PredictorCalidadAireSimple:
    """
    Versión simplificada del predictor para ejemplos y pruebas rápidas.
    Utiliza valores estadísticos promedio cuando no hay modelos disponibles.
    """
    
    def __init__(self):
        self.estadisticas_arequipa = {
            'PM10': {'promedio': 45.2, 'std': 18.7},
            'PM2.5': {'promedio': 25.8, 'std': 12.3}
        }
    
    def predecir_tendencia(self, temperatura: float = 18.0, 
                          humedad: float = 50.0, 
                          hora_dia: int = 12) -> Dict:
        """
        Predicción simplificada basada en estadísticas.
        
        Args:
            temperatura (float): Temperatura en °C
            humedad (float): Humedad relativa en %
            hora_dia (int): Hora del día (0-23)
            
        Returns:
            dict: Predicción con tendencias estadísticas
        """
        # Factores de ajuste basados en condiciones meteorológicas
        factor_temp = 1.0 + (temperatura - 18) * 0.02  # Mayor temperatura = más contaminación
        factor_humedad = 1.0 - (humedad - 50) * 0.005  # Mayor humedad = menos contaminación
        factor_hora = 1.2 if 7 <= hora_dia <= 9 or 17 <= hora_dia <= 19 else 0.8  # Horas pico
        
        factor_total = factor_temp * factor_humedad * factor_hora
        
        # Aplicar factores a estadísticas base
        pm10_valor = self.estadisticas_arequipa['PM10']['promedio'] * factor_total
        pm25_valor = self.estadisticas_arequipa['PM2.5']['promedio'] * factor_total
        
        # Agregar variabilidad aleatoria
        pm10_valor += np.random.normal(0, self.estadisticas_arequipa['PM10']['std'] * 0.3)
        pm25_valor += np.random.normal(0, self.estadisticas_arequipa['PM2.5']['std'] * 0.3)
        
        # Asegurar valores positivos
        pm10_valor = max(0, pm10_valor)
        pm25_valor = max(0, pm25_valor)
        
        # Clasificar
        def clasificar_pm(valor, tipo):
            if tipo == 'PM10':
                if valor <= 50: return 'Bueno'
                elif valor <= 100: return 'Moderado'
                elif valor <= 167: return 'Dañino para Sensibles'
                else: return 'Dañino'
            else:  # PM2.5
                if valor <= 25: return 'Bueno'
                elif valor <= 50: return 'Moderado'
                elif valor <= 75: return 'Dañino para Sensibles'
                else: return 'Dañino'
        
        resultado = {
            'predicciones': {
                'PM10': {
                    'valor': round(pm10_valor, 1),
                    'unidad': 'μg/m³',
                    'clasificacion': clasificar_pm(pm10_valor, 'PM10')
                },
                'PM2.5': {
                    'valor': round(pm25_valor, 1),
                    'unidad': 'μg/m³',
                    'clasificacion': clasificar_pm(pm25_valor, 'PM2.5')
                }
            },
            'alertas': []
        }
        
        # Agregar alertas si es necesario
        if pm10_valor > 100 or pm25_valor > 50:
            resultado['alertas'].append({
                'tipo': 'CALIDAD_AIRE_ELEVADA',
                'mensaje': 'Niveles de contaminación por encima de lo recomendado'
            })
        
        return resultado


# Función de utilidad para uso directo
def predecir_calidad_aire_arequipa(datos: Dict) -> Dict:
    """
    Función utilitaria para realizar predicción directa.
    
    Args:
        datos (dict): Datos meteorológicos
        
    Returns:
        dict: Resultado de la predicción
    """
    predictor = PredictorCalidadAire()
    return predictor.predecir(datos)


# Información del módulo
__version__ = "1.0.0"
__author__ = "Sistema de Monitoreo Ambiental"
__description__ = "Predictor de calidad del aire para Arequipa, Perú"

# Exportar clases principales
__all__ = [
    'PredictorCalidadAire',
    'PredictorCalidadAireSimple', 
    'predecir_calidad_aire_arequipa'
]


if __name__ == "__main__":
    # Ejemplo de uso directo
    print("🌍 Predictor de Calidad del Aire - Arequipa")
    print("=" * 50)
    
    # Probar predictor principal
    predictor = PredictorCalidadAire()
    
    if predictor.cargado:
        datos_prueba = {
            'FECHA_INICIO': '2021-12-15 14:00:00',
            'TEMPERATURA': 22.5,
            'HUMEDAD_RELATIVA': 40.0,
            'PRESION_BAROMETRICA': 745.0
        }
        
        resultado = predictor.predecir(datos_prueba)
        print(f"\n📊 RESULTADO DE PREDICCIÓN:")
        print(f"PM10: {resultado['predicciones']['PM10']['valor']} μg/m³ ({resultado['predicciones']['PM10']['clasificacion']})")
        print(f"PM2.5: {resultado['predicciones']['PM2.5']['valor']} μg/m³ ({resultado['predicciones']['PM2.5']['clasificacion']})")
        
        if resultado.get('alertas'):
            print(f"\n🚨 ALERTAS: {len(resultado['alertas'])}")
            for alerta in resultado['alertas']:
                print(f"   • {alerta['mensaje']}")
    else:
        print("\n⚠️ Modelos no disponibles, usando predictor simplificado...")
        predictor_simple = PredictorCalidadAireSimple()
        resultado = predictor_simple.predecir_tendencia(
            temperatura=22.5, 
            humedad=40.0, 
            hora_dia=14
        )
        print(f"\n📊 PREDICCIÓN ESTADÍSTICA:")
        print(f"PM10: {resultado['predicciones']['PM10']['valor']} μg/m³")
        print(f"PM2.5: {resultado['predicciones']['PM2.5']['valor']} μg/m³")
