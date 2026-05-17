import os
import cv2
import numpy as np
import pyvista as pv
from math import radians, cos, sin

# === Paramètres ===
folder = "Samples_reframed"
angles_deg = [0, 36, 72]           # Angles entre les caméras
radius = 100                       # Distance caméra-centre (mm)
volume_resolution = 100            # Résolution du volume 3D
voxel_size = 2                     # Taille d’un voxel en mm

# === Chargement et seuillage des silhouettes ===
silhouettes = []
for i in range(3):
    path = os.path.join(folder, f"reframed_2015.06.20_09.25.13_flake_47090_cam_{i}.png")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    silhouettes.append(binary)

# === Initialisation du volume voxelisé ===
volume = np.ones((volume_resolution, volume_resolution, volume_resolution), dtype=bool)
grid = np.linspace(-radius, radius, volume_resolution)
X, Y, Z = np.meshgrid(grid, grid, grid, indexing='ij')

# === Volume carving à partir des silhouettes ===
for i, silhouette in enumerate(silhouettes):
    angle = radians(angles_deg[i])
    cx, cy = radius * cos(angle), radius * sin(angle)
    cam_pos = np.array([cx, cy, 0])
    cam_dir = np.array([-cos(angle), -sin(angle), 0])

    img_h, img_w = silhouette.shape
    f = img_w / 2  # Focale virtuelle

    for xi in range(volume_resolution):
        for yi in range(volume_resolution):
            for zi in range(volume_resolution):
                point = np.array([X[xi, yi, zi], Y[xi, yi, zi], Z[xi, yi, zi]])
                ray = point - cam_pos
                ray_proj = ray - np.dot(ray, cam_dir) * cam_dir

                u = int(f + (ray_proj[0] * f / radius))
                v = int(f + (ray_proj[1] * f / radius))

                if 0 <= u < img_w and 0 <= v < img_h:
                    if silhouette[v, u] == 0:
                        volume[xi, yi, zi] = False
                else:
                    volume[xi, yi, zi] = False

# === Création du maillage 3D PyVista (UniformGrid)
dims = np.array(volume.shape) + 1
grid3d = pv.UniformGrid()
grid3d.dimensions = dims
grid3d.spacing = (voxel_size, voxel_size, voxel_size)
grid3d.origin = (-radius, -radius, -radius)

# === Affectation des scalaires aux points (0/1) à partir du volume
scalars = np.zeros(dims, dtype=np.uint8)
scalars[:-1, :-1, :-1] = volume.astype(np.uint8)
grid3d.point_data["values"] = scalars.flatten(order="F")

# === Extraction de l'isosurface
contour = grid3d.contour(isosurfaces=[0.5], scalars="values")

# === Affichage interactif
plotter = pv.Plotter()
plotter.add_mesh(contour, color="white", show_edges=False)
plotter.add_axes()
plotter.show()

# === Export STL
stl_path = os.path.join(folder, "flake_3D_reconstruction.stl")
contour.save(stl_path)
print(f"✅ Export STL terminé : {stl_path}")
