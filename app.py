import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Configuración de página
st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")

# 1. Cargar el modelo
@st.cache_resource
def load_my_model():
    return load_model("best_model.keras")

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# 2. Interfaz
st.title("🔢 Clasificador de Números MNIST")
st.write("Sube una imagen de un número escrito a mano sobre fondo blanco o negro.")

uploaded_file = st.file_uploader("Elige una imagen...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Procesar imagen
    image = Image.open(uploaded_file).convert('L') # Escala de grises
    
    # IMPORTANTE: MNIST espera fondo negro y trazo blanco. 
    # Si la imagen tiene fondo blanco, la invertimos.
    # Calculamos el promedio de brillo para decidir.
    if np.mean(image) > 127:
        image = ImageOps.invert(image)

    # Redimensionar
    image = image.resize((28, 28))
    
    # Preprocesamiento IGUAL al entrenamiento
    img_array = np.array(image).astype('float32')
    img_array /= 255.0  # Escalar a [0, 1]
    img_array -= 0.5    # Desplazar a [-0.5, 0.5] como en el notebook
    
    # Ajustar forma a (1, 784) porque el modelo tiene Flatten en la entrada de un vector
    img_input = img_array.reshape(1, 784)

    # Mostrar imagen procesada
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Vista para el modelo (28x28)", width=150)

    # Predicción
    prediction = model.predict(img_input)
    resultado = np.argmax(prediction)
    confianza = np.max(prediction) * 100

    with col2:
        st.metric("Número Detectado", f"{resultado}")
        st.write(f"Confianza: {confianza:.2f}%")

    # Mostrar gráfico de barras de probabilidades
    st.bar_chart(prediction[0])
