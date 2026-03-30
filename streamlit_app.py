import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import onnxruntime as ort

st.set_page_config(page_title="Brain Tumor Demo", layout="centered")

st.title("🧠 Brain Tumor Detection Demo")
st.write("Sube una imagen para ejecutar inferencia con ONNX y ver la detección")

MODEL_PATH = "best.onnx"
CONF_THRESHOLD = 0.25  # puedes subirlo a 0.40 o 0.50 si salen muchas cajas

@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_PATH)
    return session

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    original_image = image.copy()
    original_size = original_image.size  # (width, height)

    image_resized = image.resize((640, 640))
    img_array = np.array(image_resized).astype(np.float32) / 255.0

    # HWC -> CHW
    img_array = np.transpose(img_array, (2, 0, 1))

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    return original_image, original_size, img_array

def draw_detections(image, detections, original_size):
    draw = ImageDraw.Draw(image)
    orig_w, orig_h = original_size

    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0

    count = 0

    for det in detections:
        x1, y1, x2, y2, conf, cls = det

        if conf < CONF_THRESHOLD:
            continue

        # escalar coordenadas desde 640x640 a tamaño original
        x1 = x1 * scale_x
        x2 = x2 * scale_x
        y1 = y1 * scale_y
        y2 = y2 * scale_y

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label = f"tumor? {conf:.2f} | cls {int(cls)}"
        draw.text((x1, max(0, y1 - 15)), label, fill="red")
        count += 1

    return image, count

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen original", use_container_width=True)

    if st.button("Detectar regiones"):
        try:
            session = load_model()

            input_name = session.get_inputs()[0].name
            output_names = [output.name for output in session.get_outputs()]

            original_image, original_size, input_tensor = preprocess_image(image)

            outputs = session.run(output_names, {input_name: input_tensor})

            # salida esperada: (1, 300, 6)
            detections = outputs[0][0]

            detected_image, num_boxes = draw_detections(
                original_image.copy(),
                detections,
                original_size
            )

            st.success("Inferencia ejecutada correctamente")
            st.write(f"Detecciones con confianza > {CONF_THRESHOLD}: {num_boxes}")

            st.image(
                detected_image,
                caption="Imagen con regiones detectadas",
                use_container_width=True
            )

            with st.expander("Ver detalles del output"):
                st.write(f"Input name: {input_name}")
                st.write(f"Input shape esperada: {session.get_inputs()[0].shape}")
                st.write(f"Número de salidas: {len(outputs)}")
                st.write(f"Shape salida principal: {outputs[0].shape}")
                st.write(detections[:10])

        except Exception as e:
            st.error(f"Error ejecutando el modelo: {e}")