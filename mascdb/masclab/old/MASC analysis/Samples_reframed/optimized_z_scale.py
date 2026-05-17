import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize_scalar
import open3d as o3d
import plotly.graph_objects as go
import os

# === Paramètres ===
THRESHOLD = 10
IMG_FILES = [
    ("reframed_2015.06.20_09.25.13_flake_47090_cam_0.png", 0),
    ("reframed_2015.06.20_09.25.13_flake_47090_cam_1.png", 36),
    ("reframed_2015.06.20_09.25.13_flake_47090_cam_2.png", 72),
]
NEIGHBOR_RADIUS = 0.9

def generate_point_clouds(z_scale):
    clouds = []
    for img_path, angle_deg in IMG_FILES:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = img > THRESHOLD
        ys, xs = np.where(mask)
        intens = img[ys, xs].astype(np.float32)

        y_c, x_c = ys.mean(), xs.mean()
        xs_rel = xs - x_c
        ys_rel = ys - y_c
        #zs = intens * z_scale
        zs = (255 - intens) * z_scale
        
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = xs_rel * cos_t - zs * sin_t
        y_rot = ys_rel
        z_rot = xs_rel * sin_t + zs * cos_t

        pts = np.stack([x_rot, y_rot, z_rot], axis=1)
        clouds.append(pts)

    return clouds

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

z_scores = []
overlap_scores = []

def cost_function(z_scale):
    clouds = generate_point_clouds(z_scale)
    score = compute_overlap_score(clouds)
    z_scores.append(z_scale)
    overlap_scores.append(score)
    return -score

# === Optimisation du Z_SCALE
result = minimize_scalar(cost_function, bounds=(0.01, 1.0), method='bounded')
optimal_z = result.x

# === Courbe score
plt.figure()
plt.plot(z_scores, overlap_scores, marker='o')
plt.title("Score de recouvrement vs Z_SCALE")
plt.xlabel("Z_SCALE")
plt.ylabel("Score de recouvrement")
plt.grid(True)
plt.savefig("courbe_recouvrement.png")
plt.close()

# === Génération nuage 3D optimal
clouds = generate_point_clouds(optimal_z)
all_points = np.vstack(clouds)

# === Sauvegarde du nuage en interactif (HTML)
fig = go.Figure(data=[go.Scatter3d(
    x=all_points[:, 0],
    y=all_points[:, 1],
    z=all_points[:, 2],
    mode='markers',
    marker=dict(size=2, color=all_points[:, 2], colorscale='Viridis')
)])
fig.update_layout(scene=dict(
    xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
    title="Nuage de points 3D optimisé"
)
fig.write_html("nuage_points_3D.html")

# === Nettoyage du nuage de points avec Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)
pcd.estimate_normals()
pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

# === Reconstruction du mesh (Poisson Surface Reconstruction)
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_clean, depth=9)
mesh.compute_vertex_normals()
mesh = mesh.remove_degenerate_triangles()
mesh = mesh.remove_duplicated_triangles()
mesh = mesh.remove_duplicated_vertices()
mesh = mesh.remove_non_manifold_edges()

# === Affichage interactif et sauvegarde STL
o3d.visualization.draw_geometries([mesh], window_name="Mesh 3D flocon optimisé")
o3d.io.write_triangle_mesh("mesh_flocon_optimal.stl", mesh)

print(f"✅ Z_SCALE optimal trouvé : {optimal_z:.4f}")
print("✅ Mesh sauvegardé sous mesh_flocon_optimal.stl")
print("✅ Nuage 3D enregistré sous nuage_points_3D.html")
print("✅ Courbe de score : courbe_recouvrement.png")
