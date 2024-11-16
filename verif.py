from PIL import Image

# Charger les images
image1 = Image.open("line_5.png")
image2 = Image.open("line_5_zoomed.png")

# Afficher les dimensions
print("Dimensions de la première image (zoom 10) :", image1.size)
print("Dimensions de la deuxième image (zoom 5) :", image2.size)
