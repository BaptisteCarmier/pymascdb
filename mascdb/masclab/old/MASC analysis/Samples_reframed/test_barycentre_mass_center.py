import numpy as np
import cv2
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

# === Paramètres ===
THRESHOLD = 10
IMG_FILES = [
    ("reframed_2015.06.20_09.25.13_flake_47090_cam_0.png", 0),
    ("reframed_2015.06.20_09.25.13_flake_47090_cam_1.png", 36),
    ("reframed_2015.06.20_09.25.13_flake_47090_cam_2.png", 72),
]
Z_SCALE = 0.45  # Ajusté

point_clouds = []
axis_lines = []

plt.figure(figsize=(12, 4))  # Pour affichage 2D

for idx, (img_path, angle_deg) in enumerate(IMG_FILES):
    if not os.path.exists(img_path):
        print(f"Image non trouvée : {img_path}")
        continue

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erreur lecture image : {img_path}")
        continue

    mask = img > THRESHOLD
    ys, xs = np.where(mask)
    intens = img[ys, xs].astype(np.float32)

    y_c, x_c = ys.mean(), xs.mean()
    xs_rel = xs - x_c
    ys_rel = ys - y_c
    zs = intens * Z_SCALE  # Profondeur inversée

    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot = xs_rel * cos_t - zs * sin_t
    y_rot = ys_rel
    z_rot = xs_rel * sin_t + zs * cos_t

    pts = np.stack([x_rot, y_rot, z_rot], axis=1)
    point_clouds.append(pts)

    # Axe de rotation (vertical Y)
    axis_x = np.array([0, 0])
    axis_y = np.array([-img.shape[0] / 2, img.shape[0] / 2])
    axis_z = np.array([0, 0])
    axis = np.stack([axis_x, axis_y, axis_z], axis=1)
    axis_rot = axis @ np.array([
        [cos_t, 0, -sin_t],
        [0,     1,  0],
        [sin_t, 0,  cos_t]
    ]).T
    axis_lines.append(axis_rot)

    # Affichage image avec barycentre + axe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    x_c_int, y_c_int = int(round(x_c)), int(round(y_c))
    img_rgb = cv2.circle(img_rgb, (x_c_int, y_c_int), 5, (0, 255, 0), -1)
    img_rgb = cv2.line(img_rgb, (x_c_int, 0), (x_c_int, img.shape[0]-1), (0, 255, 0), 1)

    plt.subplot(1, 3, idx + 1)
    plt.imshow(img_rgb)
    plt.title(f"cam_{idx} - θ={angle_deg}°")
    plt.axis("off")

plt.suptitle("Images avec axe de rotation et barycentre")
plt.tight_layout()
plt.savefig("axes_et_barycentres.png", dpi=300)
plt.show()

# === Nuage de points combiné
all_points = np.vstack(point_clouds)

# === Calcul centre de masse global
center_mass = all_points.mean(axis=0)

# === Tracé 3D interactif
fig = go.Figure()

# Points
fig.add_trace(go.Scatter3d(
    x=all_points[:, 0],
    y=all_points[:, 1],
    z=all_points[:, 2],
    mode='markers',
    marker=dict(size=2, color=all_points[:, 2], colorscale='Viridis'),
    name='Points'
))

# Axes de rotation
for i, axis in enumerate(axis_lines):
    fig.add_trace(go.Scatter3d(
        x=axis[:, 0],
        y=axis[:, 1],
        z=axis[:, 2],
        mode='lines',
        line=dict(color='green', width=6),
        name=f"Axe cam_{i}"
    ))

# Centre de masse global
fig.add_trace(go.Scatter3d(
    x=[center_mass[0]],
    y=[center_mass[1]],
    z=[center_mass[2]],
    mode='markers',
    marker=dict(size=6, color='red'),
    name="Centre de masse"
))

fig.update_layout(
    scene=dict(
        xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
        aspectmode='data'
    ),
    title="Représentation 3D des flocons, axes et barycentre"
)

fig.write_html("flocons_axes_3d.html")
fig.show()
