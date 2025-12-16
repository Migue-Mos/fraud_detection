# Detección de Fraude en Transacciones

## Descripción del Proyecto
El objetivo de este proyecto es detectar transacciones fraudulentas utilizando Machine Learning. Esto permite identificar comportamientos sospechosos y mitigar riesgos financieros para la empresa.

## Dataset
- Fuente: Kaggle - Credit Card Fraud Detection Dataset
- Variables: V1 a V28 (features anonimizadas), Amount, Class
- Variable objetivo: `Class` (0 = No fraude, 1 = Fraude)

## Estructura del Proyecto

fraud_detection/
├── data/             # Dataset original y limpio para pruebas
├── src/              # Código fuente: carga, preprocesamiento, modelado, evaluación
├── notebooks/        # Análisis exploratorio (EDA) y modelado
├── models/           # Modelos entrenados y scaler
├── results/          # Plots y métricas generadas
├── main.py           # Script principal para ejecutar pipeline completo
├── README.md         # Este archivo
├── requirements.txt  # Dependencias del proyecto

## Preprocesamiento de Datos
- Eliminar valores nulos  
- Escalado de variables numéricas (StandardScaler)  
- Separación de features y target (`Class`)  
- División en conjunto de entrenamiento y test

## Modelos Entrenados
- **Regresión Logística**: lineal y fácilmente interpretable  
- **Random Forest**: robusto, maneja relaciones no lineales  
- **SVM lineal**: eficiente para datos escalados

## Métricas de Evaluación
- Accuracy  
- Precision, Recall, F1-score  
- ROC-AUC  
- Matriz de Confusión

## Conclusiones Clave
- Random Forest es el modelo más robusto en este dataset  
- Variables más importantes según Random Forest: V11, V4, V2, V21  
- Los pasos de limpieza y escalado son críticos para rendimiento del modelo

## Recomendaciones
- Monitorizar transacciones sospechosas automáticamente  
- Revisar alertas basadas en las variables más relevantes  
- Continuar recolectando datos para mejorar modelos

## Cómo Ejecutar
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo
python main.py

