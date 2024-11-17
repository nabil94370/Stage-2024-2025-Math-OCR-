import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Activer le fallback pour MPS si nécessaire

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

# Définir les dimensions maximales d'affichage des images
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
def get_page_image(pdf_file, page_num, dpi=96):
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    return png.convert("RGB")  # Convertir en RGB sans appliquer de zoom


@st.cache_data()
def get_uploaded_image(in_file):
    return Image.open(in_file).convert("RGB")  # Charger l'image en RGB sans appliquer de zoom


def resize_image(pil_image):
    if pil_image is None:
        return
    pil_image.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.Resampling.LANCZOS)


@st.cache_data()
def page_count(pdf_file):
    doc = open_pdf(pdf_file)
    return len(doc)


def get_canvas_hash(pil_image):
    return hashlib.md5(pil_image.tobytes()).hexdigest()


@st.cache_data()
def get_image_size(pil_image):
    if pil_image is None:
        return MAX_HEIGHT, MAX_WIDTH
    height, width = pil_image.height, pil_image.width
    return height, width


st.set_page_config(layout="wide")

top_message = """### Texify

After the model loads, upload an image, a PDF, or select a folder of images. Then, draw a box around the equation or text you want to OCR by clicking and dragging. Alternatively, process all images in a folder at once.
"""

st.markdown(top_message)
col1, col2 = st.columns([.7, .3])

model = load_model_cached()
processor = load_processor_cached()

# Sidebar: select input mode
input_mode = st.sidebar.radio("Input Mode", ["Single Image/PDF", "Folder of Images"])
temperature = st.sidebar.slider("Generation temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

if input_mode == "Single Image/PDF":
    in_file = st.sidebar.file_uploader("PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"])
    if in_file is None:
        st.stop()

    filetype = in_file.type
    if "pdf" in filetype:
        page_count = page_count(in_file)
        page_number = st.sidebar.number_input(f"Page number out of {page_count}:", min_value=1, value=1, max_value=page_count)
        pil_image = get_page_image(in_file, page_number)
    else:
        pil_image = get_uploaded_image(in_file)

    resize_image(pil_image)

    canvas_hash = get_canvas_hash(pil_image) if pil_image else "canvas"

    with col1:
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.1)",  # Fixed fill color with some opacity
            stroke_width=1,
            stroke_color="#FFAA00",
            background_color="#FFF",
            background_image=pil_image,
            update_streamlit=True,
            height=get_image_size(pil_image)[0],
            width=get_image_size(pil_image)[1],
            drawing_mode="rect",
            point_display_radius=0,
            key=canvas_hash,
        )

    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        bbox_list = None
        if objects.shape[0] > 0:
            boxes = objects[objects["type"] == "rect"][["left", "top", "width", "height"]]
            boxes["right"] = boxes["left"] + boxes["width"]
            boxes["bottom"] = boxes["top"] + boxes["height"]
            bbox_list = boxes[["left", "top", "right", "bottom"]].values.tolist()

        if bbox_list:
            with col2:
                inferences = [infer_image(pil_image, bbox, temperature) for bbox in bbox_list]
                for idx, inference in enumerate(reversed(inferences)):
                    st.markdown(f"### {len(inferences) - idx}")
                    katex_markdown = replace_katex_invalid(inference)
                    st.markdown(katex_markdown)
                    st.code(inference)
                    st.divider()

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
