import cv2
import numpy as np
import os

def split_image_into_phrases_with_zoom_and_padding(image_path, output_dir, zoom_factor=3.0, padding=10):
    # Charger l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image introuvable. Veuillez vérifier le chemin.")
        return

    # Binarisation de l'image
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Trouver les contours pour détecter les blocs de texte
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Trier les contours par leur position de gauche à droite et de haut en bas
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    # Fusionner les contours en blocs de phrases
    phrase_bounds = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not phrase_bounds:
            phrase_bounds.append((x, y, x + w, y + h))
        else:
            px1, py1, px2, py2 = phrase_bounds[-1]
            # Fusionner si les contours sont proches verticalement et horizontalement
            if abs(y - py1) < 15:  # Ajustez ce seuil si nécessaire
                phrase_bounds[-1] = (min(px1, x), min(py1, y), max(px2, x + w), max(py2, y + h))
            else:
                phrase_bounds.append((x, y, x + w, y + h))

    # Créer un dossier pour les sorties
    os.makedirs(output_dir, exist_ok=True)

    # Découper, ajouter du padding, et sauvegarder chaque phrase avec un zoom
    for idx, (x1, y1, x2, y2) in enumerate(phrase_bounds):
        # Ajouter du padding
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(img.shape[1], x2 + padding)
        y2_padded = min(img.shape[0], y2 + padding)

        # Découpe avec padding
        phrase_img = img[y1_padded:y2_padded, x1_padded:x2_padded]

        # Appliquer le zoom
        zoomed_img = cv2.resize(phrase_img, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

        # Sauvegarder l'image zoomée au format JPG
        output_path = os.path.join(output_dir, f"phrase_{idx}.jpg")
        cv2.imwrite(output_path, zoomed_img, [cv2.IMWRITE_JPEG_QUALITY, 95])  # 95 = qualité JPG

    print(f"Image divisée en {len(phrase_bounds)} phrases avec padding et un zoom de {zoom_factor*100}%. Résultats enregistrés en JPG dans {output_dir}.")

# Exemple d'utilisation
split_image_into_phrases_with_zoom_and_padding("moin.jpg", "output_phrases", zoom_factor=3.0, padding=10)
