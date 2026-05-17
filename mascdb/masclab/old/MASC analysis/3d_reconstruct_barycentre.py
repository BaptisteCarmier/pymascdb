import os
import cv2
import numpy as np
from collections import Counter
from scipy.spatial import Delaunay
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import open3d as o3d

# === PARAMÈTRES ===
IMAGE_FILE = "highlighted_meteor.png"
THRESH_VALUE = 10
Z_SCALE = 0.25
FUSION_RADIUS = 2.0
CAMERA_ANGLES = [0, 36, 72]
EXPORT_STL = "flake_clean.stl"

# === ÉTAPE 1 : CHARGEMENT + DÉCOUPE ===
img_rgb = cv2.cvtColor(cv2.imread(IMAGE_FILE), cv2.COLOR_BGR2RGB)
h, w, _ = img_rgb.shape
slice_w = w // 3
slices = [img_rgb[:, i * slice_w:(i + 1) * slice_w] for i in range(3)]

point_list = []
intensity_list = []

for idx, (slc, angle_deg) in enumerate(zip(slices, CAMERA_ANGLES)):
    gray = cv2.cvtColor(slc, cv2.COLOR_RGB2GRAY)
    light_bg_mask = gray > 220  # filtre fond clair (blanc ou gris clair)

    red_mask = (slc[:, :, 0] > 200) & (slc[:, :, 1] < 80) & (slc[:, :, 2] < 80)

    obj_mask = ~(light_bg_mask | red_mask)

    ys, xs = np.where(obj_mask)
    if len(xs) == 0:
        continue

    y_c, x_c = ys.mean(), xs.mean()
    xs_rel = (xs - x_c).astype(np.float32)
    ys_rel = (ys - y_c).astype(np.float32)
    intens = gray[ys, xs].astype(np.float32)
    zs = intens * Z_SCALE

    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot = xs_rel * cos_t - zs * sin_t
    y_rot = ys_rel
    z_rot = xs_rel * sin_t + zs * cos_t

    point_list.append(np.stack([x_rot, y_rot, z_rot], axis=1))
    intensity_list.append(intens)

points = np.vstack(point_list)
intensities = np.hstack(intensity_list)

# === ÉTAPE 2 : VISU INTERACTIVE
traces = []
colors = ['blue', 'green', 'purple']
for i, pts in enumerate(point_list):
    traces.append(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(size=1.5, color=colors[i]),
        name=f"cam_{i}"
    ))
traces.append(go.Scatter3d(
    x=[0, 0], y=[-150, 150], z=[0, 0],
    mode='lines', line=dict(color='red', width=4), name='axe Y'
))
fig = go.Figure(traces)
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
fig.write_html("flocon_clean_interactif.html")
print("➡️  Vue 3D enregistrée : flocon_clean_interactif.html")

# === ÉTAPE 3 : FILTRAGE BRUIT
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
points_filt = np.asarray(pcd_clean.points)
intens_filt = intensities[ind]

# === ÉTAPE 4 : FUSION AVANCÉE
nn = NearestNeighbors(radius=FUSION_RADIUS).fit(points_filt)
visited = set()
fused_pts = []

for i, pt in enumerate(points_filt):
    if i in visited:
        continue
    idx = nn.radius_neighbors([pt], return_distance=False)[0]
    cluster = list(set(idx) - visited)
    if not cluster:
        continue
    cluster_pts = points_filt[cluster]
    cluster_weights = intens_filt[cluster]
    w_sum = cluster_weights.sum()
    if w_sum > 0:
        avg_point = np.average(cluster_pts, axis=0, weights=cluster_weights)
    else:
        avg_point = np.mean(cluster_pts, axis=0)
    fused_pts.append(avg_point)
    visited.update(cluster)

fused_pts = np.vstack(fused_pts)
fused_pts = fused_pts[~np.isnan(fused_pts).any(axis=1)]

# === ÉTAPE 5 : MESH STL
scaler = StandardScaler()
scaled = scaler.fit_transform(fused_pts)
tri = Delaunay(scaled)
faces = []
for tet in tri.simplices:
    faces += [tuple(sorted(f)) for f in
              ((tet[0], tet[1], tet[2]),
               (tet[0], tet[1], tet[3]),
               (tet[0], tet[2], tet[3]),
               (tet[1], tet[2], tet[3]))]
counts = Counter(faces)
boundary_faces = [f for f, c in counts.items() if c == 1]

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(fused_pts)
mesh.triangles = o3d.utility.Vector3iVector(boundary_faces)
mesh.compute_vertex_normals()

o3d.io.write_triangle_mesh(EXPORT_STL, mesh)
print(f"✅ Mesh STL exporté : {EXPORT_STL}")

# === VISU OPEN3D ===
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
