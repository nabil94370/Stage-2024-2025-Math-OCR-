from skimage import io, transform

def zoom_image_skimage(image_path, zoom_factor, output_path):
    # Charger l'image
    image = io.imread(image_path)
    
    # Calculer la nouvelle taille
    new_height = int(image.shape[0] * zoom_factor)
    new_width = int(image.shape[1] * zoom_factor)
    
    # Redimensionner l'image
    zoomed_image = transform.resize(image, (new_height, new_width), anti_aliasing=True)
    
    # Sauvegarder l'image zoomée
    io.imsave(output_path, (zoomed_image * 255).astype('uint8'))  # Convertir en uint8 pour l'enregistrement
    print(f"L'image zoomée a été sauvegardée dans : {output_path}")

# Exemple d'utilisation
zoom_image_skimage("line_5.png", 2.0, "zoomed_image.jpg")  # Zoom x2
