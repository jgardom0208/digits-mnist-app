import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo con la extensión correcta
model = load_model("best_model.keras")

# Crear la interfaz de usuario
st.title("Clasificador de Números Escritos")
st.write("Sube una imagen de un número (28x28) para identificarlo.")

uploaded_file = st.file_uploader("Sube una imagen en escala de grises", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Procesar la imagen
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    
    # Preparamos los datos para el modelo (1 muestra, 28x28 píxeles)
    img_array = img_array.reshape(1, 28, 28)

    # Mostrar la imagen subida
    st.image(image, caption="Imagen cargada", width=150)

    # Predicción
    prediction = model.predict(img_array)
    
    # Definimos las etiquetas de los números
    classes = ["Cero", "Uno", "Dos", "Tres", "Cuatro", "Cinco", "Seis", "Siete", "Ocho", "Nueve"]
    
    # Obtener el resultado
    numero_detectado = np.argmax(prediction)
    st.write(f"### Predicción: {numero_detectado}")
    st.write(f"Etiqueta: {classes[numero_detectado]}")
