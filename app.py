import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo (asegúrate de que el nombre coincida con el que guardaste)
model = load_model("best_model.keras")

# Crear la interfaz de usuario
st.title("Clasificador de Números MNIST")
st.write("Sube una imagen de un número para que el modelo lo identifique.")

uploaded_file = st.file_uploader("Sube una imagen en escala de grises (28x28)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Procesar la imagen
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    
    # Ajustamos la forma para que coincida con tu modelo de Keras
    img_array = img_array.reshape(1, 28, 28)

    # Mostrar la imagen subida
    st.image(image, caption="Imagen cargada", width=150)

    # Predicción
    prediction = model.predict(img_array)
    
    # Para números, las clases son simplemente los dígitos del 0 al 9
    classes = ["Cero", "Uno", "Dos", "Tres", "Cuatro", "Cinco", "Seis", "Siete", "Ocho", "Nueve"]
    
    resultado = np.argmax(prediction)
    st.write(f"### Predicción: {resultado} ({classes[resultado]})")
