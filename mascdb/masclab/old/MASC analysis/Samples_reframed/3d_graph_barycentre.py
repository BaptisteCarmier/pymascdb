import cv2
import numpy as np
import plotly.graph_objects as go
import open3d as o3d
from sklearn.preprocessing import StandardScaler
from scipy.spatial import Delaunay
from collections import Counter
import os

# === Paramètres utilisateur ===
Z_SCALE = 0.3 #0.25
THRESHOLD = 10
CAMERA_ANGLES = [0, 36, 72]
IMG_FILES = [
    ("reframed_2015.06.20_09.25.13_flake_47090_cam_0.png", 0),
    ("reframed_2015.06.20_09.25.13_flake_47090_cam_1.png", 36),
    ("reframed_2015.06.20_09.25.13_flake_47090_cam_2.png", 72),
]
OUTPUT_HTML = "flocon_3vues_interactif.html"
OUTPUT_STL = "flocon_3vues_mesh.stl"

# === Étape 1 : Extraction des points 3D
point_list = []
intensity_list = []

for img_path, angle_deg in IMG_FILES:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = img > THRESHOLD
    ys, xs = np.where(mask)
    intens = img[ys, xs].astype(np.float32)

    # Recentrage
    y_c, x_c = ys.mean(), xs.mean()
    xs_rel = xs - x_c
    ys_rel = ys - y_c
    zs = intens * Z_SCALE
    zs = (255 - intens) * Z_SCALE

    # Rotation autour de Y
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot = xs_rel * cos_t - zs * sin_t
    y_rot = ys_rel
    z_rot = xs_rel * sin_t + zs * cos_t

    point_list.append(np.stack([x_rot, y_rot, z_rot], axis=1))
    intensity_list.append(intens)

points = np.vstack(point_list)
intensities = np.hstack(intensity_list)

# === Étape 2 : Visualisation Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=points[:, 0], y=points[:, 1], z=points[:, 2],
    mode='markers',
    marker=dict(size=2, color=intensities, colorscale='Viridis', opacity=0.8)
)])
fig.update_layout(scene=dict(
    xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
), title='Nuage de points 3D du flocon (3 vues)')
fig.write_html(OUTPUT_HTML)
print(f"✅ Nuage de points interactif : {OUTPUT_HTML}")

# === Étape 3 : Nettoyage avec Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
clean_points = np.asarray(pcd_clean.points)

# === Étape 4 : Triangulation 3D (Delaunay)
scaler = StandardScaler()
scaled_points = scaler.fit_transform(clean_points)
tri = Delaunay(scaled_points)

# Extraction faces de surface
faces = []
for tet in tri.simplices:
    faces += [tuple(sorted((tet[i], tet[j], tet[k])))
              for i, j, k in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]]

counts = Counter(faces)
surface_faces = [f for f in counts if counts[f] == 1]

# === Étape 5 : Création du mesh et export
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(clean_points)
mesh.triangles = o3d.utility.Vector3iVector(surface_faces)
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh(OUTPUT_STL, mesh)

print(f"✅ Mesh STL exporté : {OUTPUT_STL}")

# === Étape 6 : Visualisation locale
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
