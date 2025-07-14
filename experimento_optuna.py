"""
Experimento de Búsqueda de Hiperparámetros con Optuna
====================================================

Este experimento compara el rendimiento de Optuna con GridSearchCV y RandomizedSearchCV
para la búsqueda de hiperparámetros en un problema de clasificación.

Autor: Tarea 1 - Primer Parcial
Dataset: Breast Cancer Wisconsin (Diagnostic)
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# Configuración para reproducibilidad
np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def cargar_y_preparar_datos():
    """
    Carga y prepara el dataset de cáncer de mama
    """
    print("🔄 Cargando el dataset de cáncer de mama...")
    X, y = load_breast_cancer(return_X_y=True)
    
    # División en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"✅ Dataset cargado exitosamente")
    print(f"   - Muestras de entrenamiento: {X_train.shape[0]}")
    print(f"   - Muestras de prueba: {X_test.shape[0]}")
    print(f"   - Características: {X_train.shape[1]}")
    print(f"   - Clases: {len(np.unique(y))}")
    
    return X_train, X_test, y_train, y_test

def experimento_gridsearch(X_train, y_train):
    """
    Experimento con GridSearchCV (método tradicional)
    """
    print("\n🔍 Iniciando experimento con GridSearchCV...")
    
    # Definir el espacio de búsqueda
    param_grid = {
        'max_depth': [3, 6, 12, 18],
        'min_samples_leaf': [1, 2, 3],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10]
    }
    
    clf = DecisionTreeClassifier(random_state=42)
    
    # Configurar GridSearchCV
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    # Medir tiempo de ejecución
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    execution_time = time.time() - start_time
    
    print(f"✅ GridSearchCV completado en {execution_time:.2f} segundos")
    print(f"   - Mejores parámetros: {grid_search.best_params_}")
    print(f"   - Mejor puntaje CV: {grid_search.best_score_:.4f}")
    print(f"   - Combinaciones probadas: {len(grid_search.cv_results_['params'])}")
    
    return grid_search, execution_time

def experimento_randomized(X_train, y_train):
    """
    Experimento con RandomizedSearchCV
    """
    print("\n🎲 Iniciando experimento con RandomizedSearchCV...")
    
    # Definir el espacio de búsqueda
    param_dist = {
        'max_depth': [3, 6, 12, 18],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10, 15]
    }
    
    clf = DecisionTreeClassifier(random_state=42)
    
    # Configurar RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=50,  # Número de combinaciones a probar
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    # Medir tiempo de ejecución
    start_time = time.time()
    random_search.fit(X_train, y_train)
    execution_time = time.time() - start_time
    
    print(f"✅ RandomizedSearchCV completado en {execution_time:.2f} segundos")
    print(f"   - Mejores parámetros: {random_search.best_params_}")
    print(f"   - Mejor puntaje CV: {random_search.best_score_:.4f}")
    print(f"   - Combinaciones probadas: {random_search.n_iter}")
    
    return random_search, execution_time

def objective_function(trial):
    """
    Función objetivo para Optuna
    Define el espacio de búsqueda y el modelo a optimizar
    """
    # Sugerir hiperparámetros
    max_depth = trial.suggest_int('max_depth', 3, 18)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    
    # Crear el modelo con los parámetros sugeridos
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=42
    )
    
    # Evaluar el modelo con validación cruzada
    # Usamos las variables globales X_train_global y y_train_global
    scores = cross_val_score(clf, X_train_global, y_train_global, cv=5, scoring='accuracy')
    
    # Retornar el puntaje promedio
    return scores.mean()

def experimento_optuna(X_train, y_train, n_trials=100):
    """
    Experimento con Optuna (método moderno de optimización bayesiana)
    """
    print(f"\n🚀 Iniciando experimento con Optuna ({n_trials} trials)...")
    
    # Hacer las variables globales para que la función objetivo pueda acceder a ellas
    global X_train_global, y_train_global
    X_train_global = X_train
    y_train_global = y_train
    
    # Crear el estudio con sampler TPE (Tree-structured Parzen Estimator)
    study = optuna.create_study(
        direction='maximize',  # Maximizar la accuracy
        sampler=TPESampler(seed=42),
        study_name='DecisionTree_Optimization'
    )
    
    # Medir tiempo de ejecución
    start_time = time.time()
    
    # Ejecutar la optimización
    study.optimize(objective_function, n_trials=n_trials, show_progress_bar=True)
    
    execution_time = time.time() - start_time
    
    print(f"✅ Optuna completado en {execution_time:.2f} segundos")
    print(f"   - Mejores parámetros: {study.best_params}")
    print(f"   - Mejor puntaje CV: {study.best_value:.4f}")
    print(f"   - Trials ejecutados: {len(study.trials)}")
    
    # Crear el mejor modelo
    best_model = DecisionTreeClassifier(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    
    return study, best_model, execution_time

def evaluar_modelos(models_dict, X_test, y_test):
    """
    Evalúa todos los modelos en el conjunto de prueba
    """
    print("\n📊 Evaluando modelos en el conjunto de prueba...")
    
    resultados = {}
    
    for nombre, modelo in models_dict.items():
        # Hacer predicciones
        y_pred = modelo.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        resultados[nombre] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred
        }
        
        print(f"✅ {nombre}:")
        print(f"   - Exactitud: {accuracy:.4f}")
        print(f"   - F1-Score: {f1:.4f}")
    
    return resultados

def analizar_convergencia_optuna(study):
    """
    Analiza la convergencia del estudio de Optuna
    """
    print("\n📈 Análisis de convergencia de Optuna...")
    
    # Extraer valores de los trials
    trial_values = [trial.value for trial in study.trials if trial.value is not None]
    
    # Calcular el mejor valor acumulativo
    best_values = []
    current_best = -np.inf
    
    for value in trial_values:
        if value > current_best:
            current_best = value
        best_values.append(current_best)
    
    # Crear gráfico de convergencia
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(trial_values, 'b-', alpha=0.6, label='Valor del trial')
    plt.plot(best_values, 'r-', linewidth=2, label='Mejor valor acumulativo')
    plt.xlabel('Número de Trial')
    plt.ylabel('Accuracy')
    plt.title('Convergencia de Optuna')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histograma de valores
    plt.subplot(1, 2, 2)
    plt.hist(trial_values, bins=20, alpha=0.7, color='skyblue')
    plt.axvline(study.best_value, color='red', linestyle='--', 
                label=f'Mejor valor: {study.best_value:.4f}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Valores de Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optuna_convergencia.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_values

def comparar_metodos(grid_time, random_time, optuna_time, grid_score, random_score, optuna_score):
    """
    Compara los tres métodos de búsqueda de hiperparámetros
    """
    print("\n📊 COMPARACIÓN FINAL DE MÉTODOS")
    print("="*50)
    
    # Crear DataFrame para comparación
    comparacion = pd.DataFrame({
        'Método': ['GridSearchCV', 'RandomizedSearchCV', 'Optuna'],
        'Tiempo (s)': [grid_time, random_time, optuna_time],
        'Mejor CV Score': [grid_score, random_score, optuna_score],
        'Eficiencia': [grid_score/grid_time, random_score/random_time, optuna_score/optuna_time]
    })
    
    print(comparacion.to_string(index=False))
    
    # Encontrar el mejor método
    mejor_accuracy = comparacion.loc[comparacion['Mejor CV Score'].idxmax(), 'Método']
    mejor_tiempo = comparacion.loc[comparacion['Tiempo (s)'].idxmin(), 'Método']
    mejor_eficiencia = comparacion.loc[comparacion['Eficiencia'].idxmax(), 'Método']
    
    print(f"\n🏆 RESULTADOS:")
    print(f"   • Mejor accuracy: {mejor_accuracy}")
    print(f"   • Más rápido: {mejor_tiempo}")
    print(f"   • Mejor eficiencia: {mejor_eficiencia}")
    
    # Crear gráfico comparativo
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Gráfico de tiempo
    axes[0].bar(comparacion['Método'], comparacion['Tiempo (s)'], color=['skyblue', 'lightgreen', 'salmon'])
    axes[0].set_title('Tiempo de Ejecución')
    axes[0].set_ylabel('Tiempo (segundos)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Gráfico de accuracy
    axes[1].bar(comparacion['Método'], comparacion['Mejor CV Score'], color=['skyblue', 'lightgreen', 'salmon'])
    axes[1].set_title('Mejor CV Score')
    axes[1].set_ylabel('Accuracy')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Gráfico de eficiencia
    axes[2].bar(comparacion['Método'], comparacion['Eficiencia'], color=['skyblue', 'lightgreen', 'salmon'])
    axes[2].set_title('Eficiencia (Score/Tiempo)')
    axes[2].set_ylabel('Eficiencia')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('comparacion_metodos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparacion

def main():
    """
    Función principal que ejecuta todos los experimentos
    """
    print("🎯 EXPERIMENTO DE BÚSQUEDA DE HIPERPARÁMETROS CON OPTUNA")
    print("="*60)
    
    # 1. Cargar y preparar datos
    X_train, X_test, y_train, y_test = cargar_y_preparar_datos()
    
    # 2. Experimento con GridSearchCV
    grid_search, grid_time = experimento_gridsearch(X_train, y_train)
    
    # 3. Experimento con RandomizedSearchCV
    random_search, random_time = experimento_randomized(X_train, y_train)
    
    # 4. Experimento con Optuna
    study, optuna_model, optuna_time = experimento_optuna(X_train, y_train, n_trials=100)
    
    # 5. Preparar modelos para evaluación
    models = {
        'GridSearchCV': grid_search.best_estimator_,
        'RandomizedSearchCV': random_search.best_estimator_,
        'Optuna': optuna_model
    }
    
    # 6. Evaluar modelos en conjunto de prueba
    resultados = evaluar_modelos(models, X_test, y_test)
    
    # 7. Analizar convergencia de Optuna
    best_values = analizar_convergencia_optuna(study)
    
    # 8. Comparación final
    comparacion = comparar_metodos(
        grid_time, random_time, optuna_time,
        grid_search.best_score_, random_search.best_score_, study.best_value
    )
    
    # 9. Reporte final
    print("\n📋 REPORTE FINAL DEL EXPERIMENTO")
    print("="*40)
    
    print(f"\n🔍 RESULTADOS EN CONJUNTO DE PRUEBA:")
    for nombre, resultado in resultados.items():
        print(f"   {nombre}:")
        print(f"     - Accuracy: {resultado['accuracy']:.4f}")
        print(f"     - F1-Score: {resultado['f1_score']:.4f}")
    
    print(f"\n⚡ ANÁLISIS DE EFICIENCIA:")
    print(f"   - GridSearchCV: {grid_search.best_score_:.4f} en {grid_time:.2f}s")
    print(f"   - RandomizedSearchCV: {random_search.best_score_:.4f} en {random_time:.2f}s")
    print(f"   - Optuna: {study.best_value:.4f} en {optuna_time:.2f}s")
    
    print(f"\n🎯 CONCLUSIONES:")
    print(f"   • Optuna utiliza optimización bayesiana para encontrar mejores hiperparámetros")
    print(f"   • La convergencia es más inteligente que la búsqueda aleatoria")
    print(f"   • Permite explorar espacios de búsqueda más complejos")
    print(f"   • Ideal para problemas con muchos hiperparámetros")
    
    return study, comparacion, resultados

if __name__ == "__main__":
    # Ejecutar el experimento principal
    study, comparacion, resultados = main()
    
    print("\n✅ Experimento completado exitosamente!")
    print("📁 Archivos generados:")
    print("   - optuna_convergencia.png")
    print("   - comparacion_metodos.png") 