# Experimentos de Búsqueda de Hiperparámetros - Tarea 1

Este repositorio contiene los experimentos de búsqueda de hiperparámetros para la Tarea 1 del primer parcial de Aprendizaje Automático.

## 📁 Estructura del Proyecto

```
├── busqueda de hiperparametros.ipynb    # Notebook base con GridSearchCV y RandomizedSearchCV
├── experimento_optuna.py                # Experimento con Optuna
├── ejecutar_experimentos.py             # Script para ejecutar todos los experimentos
├── requirements.txt                     # Dependencias del proyecto
└── README.md                           # Este archivo
```

## 🚀 Instalación

1. **Crear un entorno virtual (recomendado):**
```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
venv\Scripts\activate     # En Windows
```

2. **Instalar las dependencias:**
```bash
pip install -r requirements.txt
```

## 📊 Descripción de los Experimentos

### 1. Notebook Base: `busqueda de hiperparametros.ipynb`
- **Métodos implementados:** GridSearchCV y RandomizedSearchCV
- **Algoritmo:** DecisionTreeClassifier
- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Características:**
  - Comparación detallada entre ambos métodos
  - Análisis de convergencia
  - Evaluación en conjunto de prueba
  - Documentación completa en español

### 2. Experimento con Optuna: `experimento_optuna.py`
- **Framework:** Optuna (optimización bayesiana moderna)
- **Algoritmo de optimización:** TPE (Tree-structured Parzen Estimator)
- **Características:**
  - Búsqueda inteligente de hiperparámetros
  - Visualización de convergencia
  - Comparación con métodos tradicionales
  - Análisis de eficiencia

#### 🔬 **¿Qué hace este experimento?**
Este experimento es una **comparación educativa** entre tres métodos de búsqueda de hiperparámetros:

1. **GridSearchCV (Método tradicional exhaustivo):**
   - Prueba TODAS las combinaciones posibles (72 combinaciones)
   - Garantiza encontrar la mejor combinación en el espacio definido
   - Muy lento para espacios grandes

2. **RandomizedSearchCV (Método tradicional aleatorio):**
   - Prueba 50 combinaciones aleatorias
   - Más rápido que GridSearch
   - Puede perderse buenas combinaciones por suerte

3. **Optuna (Método moderno con optimización bayesiana):**
   - Ejecuta 100 trials inteligentes
   - **Aprende de intentos anteriores** usando TPE
   - Sugiere parámetros prometedores basándose en resultados previos
   - Balance óptimo entre velocidad y precisión

#### 🎯 **Proceso del experimento:**
1. Carga el dataset de cáncer de mama (569 muestras, 30 características)
2. Entrena DecisionTreeClassifier con cada método
3. Mide tiempo de ejecución y precisión de cada método
4. Evalúa todos los modelos en conjunto de prueba
5. Genera visualizaciones de convergencia y comparaciones
6. Produce un reporte completo con conclusiones

#### 📈 **Resultados esperados:**
- **GridSearchCV**: Más lento pero exhaustivo
- **RandomizedSearchCV**: Más rápido pero menos preciso
- **Optuna**: Mejor balance velocidad/precisión, convergencia inteligente

## 🔧 Cómo Ejecutar los Experimentos

### Opción 1: Script Interactivo (Recomendado)
```bash
python ejecutar_experimentos.py
```
Este script te permite:
- Abrir el notebook interactivo
- Ejecutar el experimento con Optuna
- Ver archivos generados
- Verificar dependencias

### Opción 2: Notebook Interactivo
```bash
jupyter notebook "busqueda de hiperparametros.ipynb"
```
Ejecuta todas las celdas paso a paso para ver la comparación entre GridSearchCV y RandomizedSearchCV.

### Opción 3: Experimento con Optuna Directo
```bash
python experimento_optuna.py
```
Este experimento ejecutará:
- GridSearchCV (referencia)
- RandomizedSearchCV (referencia)
- Optuna con 100 trials
- Análisis comparativo completo

## 📈 Resultados Esperados

### Archivos de Salida
Los experimentos generarán los siguientes archivos:

**Experimento Optuna:**
- `optuna_convergencia.png` - Análisis de convergencia
- `comparacion_metodos.png` - Comparación entre métodos

### Métricas Evaluadas
- **Accuracy:** Exactitud en validación cruzada y conjunto de prueba
- **F1-Score:** Puntaje F1 en conjunto de prueba
- **Tiempo de ejecución:** Eficiencia computacional
- **Número de evaluaciones:** Cantidad de combinaciones probadas

## 🎯 Puntos Clave del Experimento

### Métodos Comparados
1. **GridSearchCV:** Búsqueda exhaustiva (método tradicional)
2. **RandomizedSearchCV:** Búsqueda aleatoria (método tradicional)
3. **Optuna:** Optimización bayesiana moderna

### Ventajas de Optuna
- **Algoritmo TPE avanzado:** Más eficiente que la búsqueda aleatoria
- **Interfaz moderna:** Fácil de usar y configurar
- **Visualizaciones integradas:** Análisis automático de convergencia
- **Pruning automático:** Detiene trials malos temprano
- **Escalabilidad:** Maneja espacios de búsqueda complejos

### Conclusiones Esperadas
El framework de optimización bayesiana (Optuna) generalmente:
- **Encuentra mejores hiperparámetros** que la búsqueda aleatoria
- **Es más eficiente** que la búsqueda exhaustiva
- **Converge más rápidamente** a buenas soluciones
- **Permite explorar espacios de búsqueda más complejos**

#### 🎓 **Valor educativo del experimento:**
- **Demuestra la evolución** de los métodos de ML: de exhaustivos a inteligentes
- **Compara métricas reales**: tiempo, precisión, eficiencia computacional
- **Visualiza la convergencia**: cómo Optuna aprende y mejora con cada trial
- **Proporciona evidencia empírica** de por qué los métodos bayesianos son superiores
- **Prepara para problemas reales**: donde los espacios de búsqueda son enormes

## 🛠️ Personalización

### Modificar Parámetros del Experimento
Puedes ajustar los siguientes parámetros en `experimento_optuna.py`:

```python
# Número de trials para Optuna
n_trials = 100

# Espacio de búsqueda para DecisionTree
search_space = {
    'max_depth': [3, 5, 7, 10, 12, 15, 18, 20],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'criterion': ['gini', 'entropy']
}
```

### Cambiar el Algoritmo de ML
Los experimentos están configurados para DecisionTree, pero puedes cambiar a otros algoritmos modificando las funciones objetivo.

### Agregar Nuevas Métricas
Puedes agregar nuevas métricas de evaluación modificando la función `evaluar_modelos()`.

## 🔍 Análisis de Resultados

### Interpretación de Gráficos
- **Convergencia:** Muestra cómo mejora el mejor puntaje a lo largo de los trials
- **Distribución:** Histograma de todos los puntajes obtenidos
- **Comparación:** Barras comparativas entre métodos

### Métricas de Eficiencia
- **Eficiencia = Accuracy / Tiempo**
- **Mejores parámetros:** Combinación óptima encontrada
- **Puntaje CV:** Rendimiento en validación cruzada
- **Puntaje Test:** Rendimiento en conjunto de prueba

## 📝 Notas Adicionales

### Reproducibilidad
Todos los experimentos usan semillas fijas (`random_state=42`) para garantizar resultados reproducibles.

### Recursos Computacionales
Los experimentos están optimizados para usar múltiples núcleos (`n_jobs=-1`) cuando sea posible.

### Extensiones Futuras
- Agregar más algoritmos de ML
- Implementar validación cruzada estratificada
- Incluir métricas adicionales (precision, recall, AUC)
- Probar con otros datasets

## 🤝 Contribuciones

Este proyecto es parte de una tarea académica. Las mejoras y extensiones son bienvenidas.

## 📚 Referencias

- [Optuna Documentation](https://optuna.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Breast Cancer Wisconsin Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)

---

**Autor:** Tarea 1 - Primer Parcial  
**Curso:** Aprendizaje Automático  
**Dataset:** Breast Cancer Wisconsin (Diagnostic) 