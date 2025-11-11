from cpptiff import read_tiff, write_tiff
import time
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu, threshold_multiotsu
from scipy.spatial import KDTree
import itertools


segmentation_data = pd.read_csv("DSR_20250704_Pos9_8_1_589_DAPI_item_0005_TwangSegmentation_RegionProps.csv")
seed_labels = segmentation_data["id"].to_numpy()
seed_z = segmentation_data["zpos"].to_numpy()
seed_y = segmentation_data["ypos"].to_numpy()
seed_x = segmentation_data["xpos"].to_numpy()
seed_zyx = np.vstack((seed_z, seed_y, seed_x)).T
marker_indices = seed_zyx[:, 0] < 300
seed_zyx = seed_zyx[marker_indices]


seed_kdtree = KDTree(seed_zyx)

read_im_time_start = time.time()
polyT_im = read_tiff("../data/images/raw/DSR_20250704_Pos9_8_2_590_polyT.tif", [0,300])
read_im_time_end = time.time()
print(f"Took {read_im_time_end - read_im_time_start} seconds to read tiff file.")
thresh = threshold_multiotsu(polyT_im[polyT_im > 0])[-1]
print(thresh)
exit()
polyT_im_mask = np.array(polyT_im > thresh, dtype=np.uint8)

knn_labels = np.empty(polyT_im.shape, dtype=np.uint8)
print(knn_labels)

# exit()
for z in range(0, polyT_im.shape[0]):
    for y in range(0, polyT_im.shape[1]):
        for x in range(0, polyT_im.shape[2]):
            print(f"z: {z+1} / {polyT_im.shape[0]}, y: {y+1} / {polyT_im.shape[1]}, x: {x+1} / {polyT_im.shape[2]}")
            if not polyT_im_mask[tuple([z,y,x])]:
                continue
            else:
                closest_seed = seed_kdtree.query(tuple([z,y,x]), 1)[1]
                knn_labels[tuple([z,y,x])] = closest_seed

write_tiff("knn_segmentation_output.tif", knn_labels)


