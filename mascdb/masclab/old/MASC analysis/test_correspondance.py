import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Fichiers image ===
image_files = [
    "Samples_reframed/reframed_2015.06.20_09.25.13_flake_47090_cam_0.png",
    "Samples_reframed/reframed_2015.06.20_09.25.13_flake_47090_cam_1.png",
    "Samples_reframed/reframed_2015.06.20_09.25.13_flake_47090_cam_2.png",
]

# === Détection de points caractéristiques avec ORB ===
orb = cv2.ORB_create(nfeatures=1000)
keypoints_list = []
descriptors_list = []
images_gray = []

for file in image_files:
    img_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    images_gray.append(img_gray)
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

# === Mise en correspondance brute entre img0 et img1 puis img0 et img2 ===
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_01 = bf.match(descriptors_list[0], descriptors_list[1])
matches_02 = bf.match(descriptors_list[0], descriptors_list[2])

# === Extraire les points correspondant ===
def get_matched_points(matches, kp1, kp2):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2

pts0_1, pts1 = get_matched_points(matches_01, keypoints_list[0], keypoints_list[1])
pts0_2, pts2 = get_matched_points(matches_02, keypoints_list[0], keypoints_list[2])

# === Visualisation des correspondances ===
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

img_matches_01 = cv2.drawMatches(images_gray[0], keypoints_list[0],
                                 images_gray[1], keypoints_list[1],
                                 matches_01[:30], None, flags=2)

img_matches_02 = cv2.drawMatches(images_gray[0], keypoints_list[0],
                                 images_gray[2], keypoints_list[2],
                                 matches_02[:30], None, flags=2)

axs[0].imshow(img_matches_01)
axs[0].set_title("Correspondances cam_0 <-> cam_1")
axs[0].axis('off')

axs[1].imshow(img_matches_02)
axs[1].set_title("Correspondances cam_0 <-> cam_2")
axs[1].axis('off')

plt.tight_layout()
plt.show()