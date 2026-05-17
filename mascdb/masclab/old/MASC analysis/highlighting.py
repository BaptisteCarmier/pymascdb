import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go

# === Paramètres utilisateur ===
threshold_value =10  # Valeur de seuil (0-255)
input_dir = "Samples_reframed"  # Dossier avec images recadrées
image_files = [
    "reframed_2015.06.20_09.25.13_flake_47090_cam_0.png",
    "reframed_2015.06.20_09.25.13_flake_47090_cam_1.png",
    "reframed_2015.06.20_09.25.13_flake_47090_cam_2.png"
]

# === Traitement des images ===
highlighted_images = []

x_axis = []
y_axis = []
z_axis = []

for filename in image_files:
    path = os.path.join(input_dir, filename)
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    height, width = img_gray.shape
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for y in range(height):
        for x in range(1, width):
            if img_gray[y, x] <= threshold_value :
                img_color[y, x - 1] = (0, 0, 255)  # Rouge (BGR)
            else:
                x_axis.append(x)
                y_axis.append(y)
                z_axis.append(img_gray[y,x])

    highlighted_images.append(img_color)

# === Affichage ===
plt.figure(figsize=(15, 5))
for i, img in enumerate(highlighted_images):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"cam_{i} - seuil {threshold_value}")
    plt.axis("off")
plt.suptitle("Pixels précédents marqués en rouge")
plt.tight_layout()
plt.savefig("highlighted_meteor")
plt.show()


fig = go.Figure(data=[go.Scatter3d(
    x=x_axis, y=y_axis, z=z_axis,
    mode='markers',
    marker=dict(size=4, color=z_axis, colorscale='Viridis')
)])

fig.update_layout(scene=dict(
    xaxis_title='X',
    yaxis_title='Y',
    zaxis_title='Z'
))
fig.show()  # Affichage interactif dans navigateur ou Jupyter
fig.write_html("mon_plot_interactif.html")