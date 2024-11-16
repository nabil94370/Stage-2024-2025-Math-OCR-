from PIL import Image
import os
import cv2
import numpy as np

def split_and_zoom_image_nearest_neighbor(image_path, output_dir, padding=5, zoom_factor=20):
    # Charger l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image introuvable. Veuillez vérifier le chemin.")
        return

    # Binarisation de l'image
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Projeter les pixels pour détecter les espaces horizontaux
    projection = np.sum(binary, axis=1)

    # Détecter les lignes contenant du texte
    start, end = None, None
    line_segments = []
    for i, value in enumerate(projection):
        if value > 0 and start is None:  # Début d'une nouvelle ligne
            start = i
        elif value == 0 and start is not None:  # Fin de la ligne
            end = i
            line_segments.append((start, end))
            start, end = None, None

    # Gérer la dernière ligne (si elle n'a pas de fin détectée)
    if start is not None:
        line_segments.append((start, len(projection)))

    # Créer un dossier pour les sorties
    os.makedirs(output_dir, exist_ok=True)

    # Découper, zoomer et sauvegarder chaque ligne
    height, width = img.shape
    for idx, (start, end) in enumerate(line_segments):
        # Ajouter du padding tout en évitant les dépassements
        start_with_padding = max(0, start - padding)
        end_with_padding = min(height, end + padding)

        # Découper l'image avec le padding
        line_img = img[start_with_padding:end_with_padding, :]

        # Convertir l'image en format compatible avec Pillow
        line_img_pil = Image.fromarray(line_img)

        # Appliquer un zoom sans interpolation (nearest neighbor)
        zoomed_width = int(line_img_pil.width * zoom_factor)
        zoomed_height = int(line_img_pil.height * zoom_factor)
        zoomed_img = line_img_pil.resize((zoomed_width, zoomed_height), Image.NEAREST)

        # Sauvegarder l'image zoomée
        output_path = os.path.join(output_dir, f"line_{idx}_zoomed.png")
        zoomed_img.save(output_path)

    print(f"Image divisée en {len(line_segments)} lignes avec zoom sans interpolation. Résultats enregistrés dans {output_dir}.")

# Exemple d'utilisation
split_and_zoom_image_nearest_neighbor("moin.jpg", "output_phrases", padding=5, zoom_factor=20)
