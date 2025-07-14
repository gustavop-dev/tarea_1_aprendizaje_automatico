#!/usr/bin/env python3
"""
Script Ejecutor de Experimentos de Búsqueda de Hiperparámetros
==============================================================

Este script te permite ejecutar todos los experimentos de forma fácil y ordenada.
Puedes elegir qué experimentos ejecutar desde el menú interactivo.

Autor: Tarea 1 - Primer Parcial
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def verificar_dependencias():
    """Verifica que todas las dependencias estén instaladas"""
    print("🔍 Verificando dependencias...")
    
    dependencias_requeridas = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'), 
        ('scikit-learn', 'sklearn'), 
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'), 
        ('optuna', 'optuna'), 
        ('tqdm', 'tqdm')
    ]
    
    dependencias_faltantes = []
    
    for nombre_paquete, nombre_import in dependencias_requeridas:
        try:
            __import__(nombre_import)
            print(f"✅ {nombre_paquete}")
        except ImportError:
            dependencias_faltantes.append(nombre_paquete)
            print(f"❌ {nombre_paquete} - NO INSTALADO")
    
    if dependencias_faltantes:
        print(f"\n⚠️  DEPENDENCIAS FALTANTES: {', '.join(dependencias_faltantes)}")
        print("💡 Instala las dependencias con: pip install -r requirements.txt")
        return False
    
    print("\n✅ Todas las dependencias están instaladas correctamente")
    return True

def ejecutar_notebook():
    """Ejecuta el notebook de Jupyter"""
    print("\n🚀 Ejecutando notebook de Jupyter...")
    print("📓 Abriendo: 'busqueda de hiperparametros.ipynb'")
    
    try:
        # Verificar si el notebook existe
        if not Path("busqueda de hiperparametros.ipynb").exists():
            print("❌ El archivo 'busqueda de hiperparametros.ipynb' no existe")
            return False
        
        # Abrir Jupyter Notebook
        subprocess.run(['jupyter', 'notebook', 'busqueda de hiperparametros.ipynb'], 
                      check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Error al ejecutar Jupyter Notebook")
        print("💡 Asegúrate de tener Jupyter instalado: pip install jupyter")
        return False
    except FileNotFoundError:
        print("❌ Jupyter no está instalado o no está en el PATH")
        print("💡 Instala Jupyter: pip install jupyter")
        return False

def ejecutar_optuna():
    """Ejecuta el experimento con Optuna"""
    print("\n🚀 Ejecutando experimento con Optuna...")
    print("⏳ Este experimento puede tomar varios minutos...")
    
    try:
        resultado = subprocess.run(['python', 'experimento_optuna.py'], 
                                 capture_output=True, text=True, check=True)
        print(resultado.stdout)
        print("✅ Experimento Optuna completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar experimento_optuna.py: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ El archivo 'experimento_optuna.py' no existe")
        return False



def ejecutar_todos_los_experimentos():
    """Ejecuta el experimento con Optuna"""
    print("\n🚀 Ejecutando experimento con Optuna...")
    print("⏳ Esto puede tomar varios minutos...")
    
    if ejecutar_optuna():
        print("\n🎉 ¡Experimento completado exitosamente!")
        mostrar_archivos_generados()
    else:
        print("\n⚠️  El experimento falló. Revisa los mensajes de error.")

def mostrar_archivos_generados():
    """Muestra los archivos generados por los experimentos"""
    print("\n📁 Archivos generados:")
    
    archivos_esperados = [
        "optuna_convergencia.png",
        "comparacion_metodos.png"
    ]
    
    for archivo in archivos_esperados:
        if Path(archivo).exists():
            print(f"✅ {archivo}")
        else:
            print(f"❌ {archivo} - NO ENCONTRADO")

def mostrar_menu():
    """Muestra el menú principal"""
    print("\n" + "="*60)
    print("🎯 EXPERIMENTOS DE BÚSQUEDA DE HIPERPARÁMETROS - TAREA 1")
    print("="*60)
    print("\nSelecciona una opción:")
    print("1. 📓 Abrir Notebook (GridSearchCV vs RandomizedSearchCV)")
    print("2. 🚀 Ejecutar experimento con Optuna")
    print("3. 📁 Mostrar archivos generados")
    print("4. 🔍 Verificar dependencias")
    print("5. ❌ Salir")
    print("-" * 60)

def main():
    """Función principal del script"""
    print("🎯 BIENVENIDO AL EJECUTOR DE EXPERIMENTOS DE HIPERPARÁMETROS")
    print("Este script te ayudará a ejecutar todos los experimentos de la tarea.")
    
    # Verificar dependencias al inicio
    if not verificar_dependencias():
        respuesta = input("\n¿Deseas continuar de todas formas? (s/N): ").lower()
        if respuesta != 's':
            print("👋 ¡Hasta luego! Instala las dependencias y vuelve a intentar.")
            return
    
    while True:
        mostrar_menu()
        
        try:
            opcion = input("Ingresa tu opción (1-5): ").strip()
            
            if opcion == '1':
                ejecutar_notebook()
            
            elif opcion == '2':
                ejecutar_optuna()
            
            elif opcion == '3':
                mostrar_archivos_generados()
            
            elif opcion == '4':
                verificar_dependencias()
            
            elif opcion == '5':
                print("\n👋 ¡Hasta luego! Gracias por usar el ejecutor de experimentos.")
                break
            
            else:
                print("⚠️  Opción no válida. Por favor, ingresa un número del 1 al 5.")
            
            # Pausa antes de mostrar el menú de nuevo
            input("\nPresiona Enter para continuar...")
            
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego! Interrumpido por el usuario.")
            break
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            input("Presiona Enter para continuar...")

if __name__ == "__main__":
    main() 