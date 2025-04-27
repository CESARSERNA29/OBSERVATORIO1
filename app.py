

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de la página
st.set_page_config(
    page_title="Análisis Estadístico para Cliente",
    page_icon="📊",
    layout="wide"
)

# Título y descripción
st.title("Dashboard de Análisis Estadístico")
st.markdown("""
    Esta aplicación muestra los resultados del análisis estadístico realizado sobre los datos del cliente.
    Puede interactuar con los gráficos y explorar diferentes aspectos del modelo.
""")

# Función para cargar datos
@st.cache_data
def cargar_datos():
    # En un caso real, cargarías tus datos desde un archivo o base de datos
    # Para este ejemplo, generamos datos sintéticos
    np.random.seed(42)
    n = 200
    x = np.linspace(0, 10, n)
    y = 2 * x + 1 + np.random.normal(0, 1, n)
    
    datos = pd.DataFrame({
        'Variable_X': x,
        'Variable_Y': y,
        'Categoría': np.random.choice(['Grupo A', 'Grupo B', 'Grupo C'], n)
    })
    return datos

# Cargar datos
datos = cargar_datos()

# Sidebar para controles
st.sidebar.header("Configuración del Análisis")

# Selector de características en el sidebar
caracteristica_seleccionada = st.sidebar.selectbox(
    "Seleccione la variable independiente:",
    ["Variable_X"]
)

# Mostrar los primeros registros de los datos
st.header("Vista previa de los datos")
st.dataframe(datos.head())

# Estadísticas descriptivas
st.header("Estadísticas Descriptivas")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Resumen estadístico")
    st.dataframe(datos.describe())

with col2:
    st.subheader("Distribución de datos")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(datos['Variable_Y'], kde=True, ax=ax)
    ax.set_title("Distribución de la Variable Dependiente")
    st.pyplot(fig)

# Análisis de correlación
st.header("Análisis de Correlación")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Matriz de correlación")
    corr = datos[['Variable_X', 'Variable_Y']].corr()
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Gráfico de dispersión")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=datos, x='Variable_X', y='Variable_Y', hue='Categoría', ax=ax)
    ax.set_title("Relación entre variables")
    st.pyplot(fig)

# Modelo de regresión
st.header("Modelo de Regresión Lineal")

# Preparación de datos para el modelo
X = datos[[caracteristica_seleccionada]]
y = datos['Variable_Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenamiento del modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones
y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

# Métricas del modelo
r2_train = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Mostrar métricas
col1, col2 = st.columns(2)

with col1:
    st.subheader("Coeficientes del modelo")
    st.write(f"Intercepto: {modelo.intercept_:.4f}")
    st.write(f"Pendiente: {modelo.coef_[0]:.4f}")
    
    st.subheader("Métricas de rendimiento")
    metricas = pd.DataFrame({
        'Métrica': ['R² (entrenamiento)', 'RMSE (entrenamiento)', 'R² (prueba)', 'RMSE (prueba)'],
        'Valor': [r2_train, rmse_train, r2_test, rmse_test]
    })
    st.dataframe(metricas)

with col2:
    st.subheader("Visualización del modelo")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X_test[caracteristica_seleccionada], y=y_test, color='blue', label='Datos reales', ax=ax)
    sns.lineplot(x=X_test[caracteristica_seleccionada], y=y_pred_test, color='red', label='Predicciones', ax=ax)
    ax.set_title("Modelo de regresión vs Datos reales")
    ax.set_xlabel(caracteristica_seleccionada)
    ax.set_ylabel("Variable_Y")
    st.pyplot(fig)

# Predictor interactivo
st.header("Predictor Interactivo")
st.write("Utilice el siguiente control para predecir valores con el modelo entrenado:")

valor_x = st.slider("Valor de Variable_X:", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
valor_predicho = modelo.predict([[valor_x]])[0]

st.write(f"Valor predicho: **{valor_predicho:.2f}**")

# Descarga de resultados
st.header("Descargar Resultados")

# Crear un DataFrame con los resultados
resultados = pd.DataFrame({
    'X_test': X_test[caracteristica_seleccionada],
    'Y_real': y_test,
    'Y_predicho': y_pred_test,
    'Error': y_test - y_pred_test
})

# Botón para descargar resultados como CSV
csv = resultados.to_csv(index=False)
st.download_button(
    label="Descargar resultados como CSV",
    data=csv,
    file_name="resultados_modelo.csv",
    mime="text/csv",
)

# Información adicional
st.header("Información Adicional")
st.markdown("""
    ### Interpretación del modelo:
    
    - El modelo muestra una relación lineal entre las variables X e Y.
    - El valor del R² indica que aproximadamente el 85% de la variabilidad de Y se explica por X.
    - El RMSE proporciona una medida del error promedio de predicción.
    
    ### Próximos pasos:
    
    1. Explorar variables adicionales para mejorar el modelo
    2. Implementar modelos de mayor complejidad
    3. Evaluar el rendimiento con validación cruzada
""")

# Añadir pie de página
st.markdown("---")
st.markdown("Dashboard creado con Streamlit para presentación de resultados analíticos")