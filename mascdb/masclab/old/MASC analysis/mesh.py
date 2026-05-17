import cv2
import numpy as np
import os
from scipy.spatial import Delaunay
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import open3d as o3d

# === Paramètres ===
threshold_value = 50
z_scale = 0.25
fusion_radius = 2.0
input_dir = "Samples_reframed"
image_files = [
    "reframed_2015.06.20_09.25.13_flake_47090_cam_0.png",
    "reframed_2015.06.20_09.25.13_flake_47090_cam_1.png",
    "reframed_2015.06.20_09.25.13_flake_47090_cam_2.png"
]
camera_angles_deg = [0, 36, 72]

# === Étape 1 : Extraction des points 3D avec intensité ===
all_points = []
all_intensities = []

for i, filename in enumerate(image_files):
    path = os.path.join(input_dir, filename)
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img_gray.shape

    angle_rad = np.deg2rad(camera_angles_deg[i])
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    for y in range(h):
        for x in range(w):
            val = img_gray[y, x]
            if val > threshold_value:
                z = val * z_scale
                x_rot = x * cos_theta - z * sin_theta
                y_rot = y
                z_rot = x * sin_theta + z * cos_theta
                all_points.append([x_rot, y_rot, z_rot])
                all_intensities.append(val)

points = np.array(all_points)
intensities = np.array(all_intensities)

# === Étape 2 : Nettoyage par filtrage spatial (Open3D)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

filtered_points = points[ind]
filtered_intensities = intensities[ind]

# === Étape 3 : Fusion avancée par moyenne pondérée (intensité)
neighbors = NearestNeighbors(radius=fusion_radius).fit(filtered_points)
visited = set()
fused = []

for i, pt in enumerate(filtered_points):
    if i in visited:
        continue
    indices = neighbors.radius_neighbors([pt], return_distance=False)[0]
    group = list(set(indices) - visited)

    if len(group) > 1:
        weights = filtered_intensities[group]
        weights = weights / weights.sum()
        weighted_avg = np.average(filtered_points[group], axis=0, weights=weights)
        fused.append(weighted_avg)
        visited.update(group)
    else:
        fused.append(pt)

fused_points = np.array(fused)

# === Étape 4 : Delaunay 3D + extraction des triangles de surface
scaler = StandardScaler()
scaled = scaler.fit_transform(fused_points)
tri = Delaunay(scaled)

faces = []
for tet in tri.simplices:
    faces += [
        tuple(sorted([tet[0], tet[1], tet[2]])),
        tuple(sorted([tet[0], tet[1], tet[3]])),
        tuple(sorted([tet[0], tet[2], tet[3]])),
        tuple(sorted([tet[1], tet[2], tet[3]]))
    ]

face_counts = Counter(faces)
boundary_faces = [f for f in face_counts if face_counts[f] == 1]

# === Étape 5 : Création du mesh + export STL
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(fused_points)
mesh.triangles = o3d.utility.Vector3iVector(boundary_faces)
mesh.compute_vertex_normals()

o3d.io.write_triangle_mesh("flake_fused_weighted.stl", mesh)
print("✅ Export terminé : flake_fused_weighted.stl")

# === Étape 6 : Visualisation interactive
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
