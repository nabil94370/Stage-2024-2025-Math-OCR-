import os
import io
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import hashlib
import pypdfium2
from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
from texify.output import replace_katex_invalid
from PIL import Image

MAX_WIDTH = 800
MAX_HEIGHT = 1000


@st.cache_resource()
def load_model_cached():
    return load_model()


@st.cache_resource()
def load_processor_cached():
    return load_processor()


@st.cache_data()
def infer_image(pil_image, bbox, temperature):
    input_img = pil_image.crop(bbox)
    model_output = batch_inference([input_img], model, processor, temperature=temperature)
    return model_output[0]


@st.cache_data()
def infer_images_from_folder(folder_path, temperature):
    images = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'webp')):
            try:
                img = Image.open(file_path).convert("RGB")
                images.append(img)
            except Exception as e:
                st.error(f"Erreur lors de la lecture de {file_name}: {e}")

    if not images:
        st.error("Aucune image valide trouvée dans le dossier.")
        return []

    model_outputs = batch_inference(images, model, processor, temperature=temperature)
    return model_outputs


@st.cache_data()
def get_uploaded_image(in_file):
    return Image.open(in_file).convert("RGB")


def resize_image(pil_image):
    if pil_image is None:
        return
    pil_image.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.Resampling.LANCZOS)


st.set_page_config(layout="wide")

top_message = """### Texify

After the model loads, upload an image, a PDF, or select a folder of images. Then, the OCR will process the content.
"""
st.markdown(top_message)
col1, col2 = st.columns([0.7, 0.3])

model = load_model_cached()
processor = load_processor_cached()

# Sidebar: input options
input_mode = st.sidebar.radio("Input Mode", ["Single Image/PDF", "Folder of Images"])
temperature = st.sidebar.slider("Generation temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

if input_mode == "Single Image/PDF":
    in_file = st.sidebar.file_uploader("Upload a PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"])
    if in_file is None:
        st.stop()

    pil_image = get_uploaded_image(in_file)
    resize_image(pil_image)

    with col1:
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("### OCR Output")
        inference = infer_image(pil_image, bbox=(0, 0, pil_image.width, pil_image.height), temperature=temperature)
        katex_markdown = replace_katex_invalid(inference)
        st.markdown(katex_markdown)
        st.code(inference)

elif input_mode == "Folder of Images":
    folder_path = st.sidebar.text_input("Folder Path", value="")
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        st.error("Veuillez entrer un chemin de dossier valide.")
        st.stop()

    with col2:
        st.markdown("### OCR Outputs for All Images")
        results = infer_images_from_folder(folder_path, temperature=temperature)
        combined_results = ""  # Pour regrouper toutes les réponses
        if results:
            for idx, result in enumerate(results):
                st.markdown(f"### Résultat {idx + 1}")
                katex_markdown = replace_katex_invalid(result)
                st.markdown(katex_markdown)
                st.code(result)
                combined_results += result + "\n"  # Ajouter chaque réponse à la réponse combinée

            # Affichage de la réponse combinée
            st.markdown("### Réponse combinée")
            st.text_area("Réponse combinée", combined_results, height=300)
