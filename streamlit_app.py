import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort

st.set_page_config(page_title="Brain Tumor Demo", layout="centered")

st.title("🧠 Brain Tumor Detection Demo")
st.write("Sube una imagen para ejecutar inferencia con ONNX")

MODEL_PATH = "best.onnx"

@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_PATH)
    return session

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    original_image = image.copy()

    image_resized = image.resize((640, 640))
    img_array = np.array(image_resized).astype(np.float32) / 255.0

    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    return original_image, img_array

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_container_width=True)

    if st.button("Run ONNX inference"):
        try:
            session = load_model()

            input_name = session.get_inputs()[0].name
            output_names = [output.name for output in session.get_outputs()]

            original_image, input_tensor = preprocess_image(image)

            outputs = session.run(output_names, {input_name: input_tensor})

            st.success("Inferencia ejecutada correctamente")
            st.write("### Información del modelo")
            st.write(f"**Input name:** {input_name}")
            st.write(f"**Number of outputs:** {len(outputs)}")

            for i, output in enumerate(outputs):
                st.write(f"**Output {i+1} shape:** {output.shape}")
                st.write(output)

        except Exception as e:
            st.error(f"Error ejecutando el modelo: {e}")