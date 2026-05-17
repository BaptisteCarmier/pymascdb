import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Chemins vers les images
image_paths = [
    "Samples/2015.06.20_09.33.18_flake_47431_cam_0.png",
    "Samples/2015.06.20_09.33.18_flake_47431_cam_1.png",
    "Samples/2015.06.20_09.33.18_flake_47431_cam_2.png"
]

# Création du dossier de sauvegarde
output_dir = "Samples_reframed"
os.makedirs(output_dir, exist_ok=True)

# Chargement des images en niveaux de gris
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

# Détection de la particule et récupération des boîtes englobantes
bounding_boxes = []
for img in images:
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)  # Seuillage
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        bounding_boxes.append((x, y, w, h))

# Définir la taille maximale et padding
max_width = max(b[2] for b in bounding_boxes)
max_height = max(b[3] for b in bounding_boxes)
padding = 20  # Marge supplémentaire autour du flocon

# Recadrage et sauvegarde
cropped_images = []
for i, img in enumerate(images):
    x, y, w, h = bounding_boxes[i]
    center_x = x + w // 2
    center_y = y + h // 2

    crop_x1 = max(center_x - (max_width // 2 + padding), 0)
    crop_y1 = max(center_y - (max_height // 2 + padding), 0)
    crop_x2 = crop_x1 + max_width + 2 * padding
    crop_y2 = crop_y1 + max_height + 2 * padding

    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_images.append(cropped)

    # Sauvegarde
    filename = os.path.basename(image_paths[i])
    output_path = os.path.join(output_dir, f"reframed_{filename}")
    cv2.imwrite(output_path, cropped)

# (Optionnel) Affichage des images recadrées
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, crop in enumerate(cropped_images):
    axs[i].imshow(crop, cmap='gray')
    axs[i].set_title(f"Camera {i} (Saved)")
    axs[i].axis('off')

plt.tight_layout()
plt.show()
