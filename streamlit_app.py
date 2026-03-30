import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import onnxruntime as ort

st.set_page_config(page_title="Brain Tumor Demo", layout="centered")

st.title("🧠 Brain Tumor Detection Demo")
st.write("Sube una imagen para ejecutar inferencia con ONNX y visualizar la región detectada")

MODEL_PATH = "best.onnx"
CONF_THRESHOLD = 0.25

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

def get_best_detection(detections, conf_threshold=0.25):
    """
    detections expected shape: (N, 6)
    each row: [x1, y1, x2, y2, conf, cls]
    """
    valid = [det for det in detections if det[4] > conf_threshold]

    if not valid:
        return None

    best = max(valid, key=lambda x: x[4])
    return best

def draw_best_detection(image, detection, original_size):
    draw = ImageDraw.Draw(image)
    orig_w, orig_h = original_size

    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0

    x1, y1, x2, y2, conf, cls = detection

    # Escalar coordenadas desde 640x640 al tamaño original
    x1 = x1 * scale_x
    x2 = x2 * scale_x
    y1 = y1 * scale_y
    y2 = y2 * scale_y

    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    label = f"Tumor {conf:.2f}"
    draw.text((x1, max(0, y1 - 18)), label, fill="red")

    return image

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Imagen original", use_container_width=True)

    if st.button("Detectar tumor"):
        try:
            session = load_model()

            input_name = session.get_inputs()[0].name
            output_names = [output.name for output in session.get_outputs()]

            original_image, original_size, input_tensor = preprocess_image(image)

            outputs = session.run(output_names, {input_name: input_tensor})

            # salida principal esperada: (1, 300, 6)
            detections = outputs[0][0]

            best_detection = get_best_detection(detections, CONF_THRESHOLD)

            st.write("---")
            st.subheader("Resultado del análisis")

            if best_detection is not None:
                detected_image = draw_best_detection(
                    original_image.copy(),
                    best_detection,
                    original_size
                )

                max_conf = float(best_detection[4])

                st.error(f"⚠️ Se detectó tumor")
                st.write(f"**Confianza de la detección:** {max_conf:.2f}")

                st.image(
                    detected_image,
                    caption="Imagen con la mejor región detectada",
                    use_container_width=True
                )
            else:
                st.success("✅ No se detectó tumor")
                st.image(
                    original_image,
                    caption="No se encontraron regiones con confianza suficiente",
                    use_container_width=True
                )

            with st.expander("Ver detalles técnicos"):
                st.write(f"**Input name:** {input_name}")
                st.write(f"**Input shape esperada:** {session.get_inputs()[0].shape}")
                st.write(f"**Número de salidas:** {len(outputs)}")
                st.write(f"**Shape salida principal:** {outputs[0].shape}")

                preview = detections[:10]
                st.write("**Primeras 10 detecciones crudas:**")
                st.write(preview)

        except Exception as e:
            st.error(f"Error ejecutando el modelo: {e}")