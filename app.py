import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Clasificador de Dígitos MNIST")
st.write("Sube una imagen de un número (28x28) o dibuja uno.")

# Cargar el modelo (asegúrate de subir el archivo .h5 al repo o entrenarlo al inicio)
@st.cache_resource
def load_my_model():
    # Para este ejemplo, creamos uno rápido si no existe el archivo
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

model = load_my_model()

file = st.file_uploader("Sube una imagen de un dígito", type=["png", "jpg"])

if file:
    img = Image.open(file).convert('L').resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28)
    
    prediction = model.predict(img_array)
    st.write(f"### Predicción: {np.argmax(prediction)}")
    st.bar_chart(prediction[0])
