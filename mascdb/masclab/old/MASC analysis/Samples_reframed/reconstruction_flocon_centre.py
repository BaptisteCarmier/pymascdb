import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
import open3d as o3d
import plotly.graph_objects as go
import os

# === Paramètres ===
THRESHOLD =20
IMG_FILES = [
    ("reframed_2015.06.20_09.33.18_flake_47431_cam_0.png", 0),
    ("reframed_2015.06.20_09.33.18_flake_47431_cam_1.png", 36),
    ("reframed_2015.06.20_09.33.18_flake_47431_cam_2.png", 72),
]
NEIGHBOR_RADIUS = 1.0

# === Génération des points 3D ===
def generate_point_clouds(z_scale, distances):
    clouds = []
    for idx, (img_path, angle_deg) in enumerate(IMG_FILES):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = img > THRESHOLD
        ys, xs = np.where(mask)
        intens = img[ys, xs].astype(np.float32)

        y_c, x_c = ys.mean(), xs.mean()
        xs_rel = xs - x_c
        ys_rel = ys - y_c
        zs = (255 - intens) * z_scale

        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = xs_rel * cos_t - zs * sin_t
        y_rot = ys_rel
        z_rot = xs_rel * sin_t + zs * cos_t

        # Translation le long de la normale
        x_rot += distances[idx] * sin_t
        z_rot -= distances[idx] * cos_t

        pts = np.stack([x_rot, y_rot, z_rot], axis=1)
        clouds.append(pts)

    return clouds

# === Score de recouvrement ===
def compute_overlap_score(clouds, radius=NEIGHBOR_RADIUS):
    if len(clouds) < 2:
        return 0
    total_matches = 0
    total_points = 0
    for i in range(len(clouds)):
        for j in range(i + 1, len(clouds)):
            pts1 = clouds[i]
            pts2 = clouds[j]
            nbrs = NearestNeighbors(radius=radius).fit(pts2)
            _, indices = nbrs.radius_neighbors(pts1)
            matches = sum(len(ind) > 0 for ind in indices)
            total_matches += matches
            total_points += len(pts1)
    return total_matches / total_points if total_points > 0 else 0

# === Optimisation distances et z_scale ===
def cost(params):
    z_scale = params[0]
    distances = params[1:]
    clouds = generate_point_clouds(z_scale, distances)
    return -compute_overlap_score(clouds)

x0 = [0.1, -20.0, 0.0, -30.0]
bounds = [(0.01, 0.3), (30.0, 100.0), (30.0, 100.0), (30.0, 100.0)]
res = minimize(cost, x0=x0, bounds=bounds, method="L-BFGS-B")
optimal_z = res.x[0]
optimal_d = res.x[1:]

# === Nuage de points filtré ===
clouds = generate_point_clouds(optimal_z, optimal_d)
all_points = np.vstack(clouds)

# === Filtrage par plan cam1 ===
angle_deg = IMG_FILES[1][1]
theta = np.deg2rad(angle_deg)
n = np.array([np.sin(theta), 0, -np.cos(theta)])  # normale au plan
cam1_pts = clouds[1]
barycenter = cam1_pts.mean(axis=0)

relative = all_points - barycenter
dot_prods = relative @ n
keep = dot_prods <= 10  # Garde les points du bon côté
filtered_points = all_points[keep]

# === Visualisation 3D (nuage)
fig = go.Figure(data=[go.Scatter3d(
    x=filtered_points[:, 0],
    y=filtered_points[:, 1],
    z=filtered_points[:, 2],
    mode='markers',
    marker=dict(size=2, color=filtered_points[:, 2], colorscale='Viridis')
)])
fig.update_layout(scene=dict(
    xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
    title="Nuage 3D filtré par plan Cam1"
)
fig.write_html("nuage_filtre.html")

# === Mesh 3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points)
pcd.estimate_normals()
pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_clean, depth=9)
mesh.compute_vertex_normals()
mesh = mesh.remove_degenerate_triangles()
mesh = mesh.remove_duplicated_triangles()
mesh = mesh.remove_duplicated_vertices()
mesh = mesh.remove_non_manifold_edges()

# === Sauvegardes
o3d.visualization.draw_geometries([mesh], window_name="Mesh filtré")
o3d.io.write_triangle_mesh("mesh_flocon_filtre.stl", mesh)

print(f"✅ Z_SCALE optimal : {optimal_z:.4f}")
print(f"✅ Distances cam : {optimal_d}")
print("✅ Mesh : mesh_flocon_filtre.stl")
print("✅ Nuage : nuage_filtre.html")