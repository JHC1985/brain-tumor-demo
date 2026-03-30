import streamlit as st
from PIL import Image

st.set_page_config(page_title="Brain Tumor Demo", layout="centered")

st.title("🧠 Brain Tumor Detection Demo")
st.write("Sube una imagen para probar la interfaz")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_container_width=True)
    st.success("Imagen cargada correctamente")