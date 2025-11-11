from cpptiff import read_tiff, write_tiff
import time
import numpy as np
import pandas as pd
from skimage.segmentation import watershed
from skimage.morphology import ball
from skimage.filters import threshold_otsu, rank

read_im_time_start = time.time()
segmentation_data = pd.read_csv("DSR_20250704_Pos9_8_1_589_DAPI_item_0005_TwangSegmentation_RegionProps.csv")

polyT_im = read_tiff("Dark.tif", [0,300])
polyT_dt = read_tiff("../data/images/watershed/DSR_20250704_Pos9_8_2_590_polyT_Dark_0_300_DT.tif")
read_im_time_end = time.time()
print(f"Took {read_im_time_end - read_im_time_start} seconds to read tiff file.")
# print(polyT_im[0][1])

radius = 15
selem = ball(radius)
mask = polyT_im > 0

print(polyT_im.shape)
print(selem.shape)
print(mask.shape)

polyT_im_mask = rank.otsu(polyT_im, selem, mask=mask)

# thresh = threshold_otsu(polyT_im[polyT_im > 0])
# print(thresh)
# polyT_im_mask = np.array(polyT_im > thresh, dtype=np.uint8)

seed_labels = segmentation_data["id"].to_numpy()
seed_z = segmentation_data["zpos"].to_numpy()
seed_y = segmentation_data["ypos"].to_numpy()
seed_x = segmentation_data["xpos"].to_numpy()
seed_zyx = np.vstack((seed_z, seed_y, seed_x)).T

marker_indices = seed_zyx[:, 0] < 300
seed_zyx = seed_zyx[marker_indices]
seed_markers = np.empty(polyT_im.shape, dtype=int)
seed_markers[tuple(seed_zyx.T)] = seed_labels[marker_indices]


watershed_time_start = time.time()
labels = watershed(-polyT_dt, markers=seed_markers, mask=polyT_im_mask)
watershed_time_end = time.time()
print(f"Took {watershed_time_end - watershed_time_start} seconds to perform watershed segmentation.")


write_tiff("watershed_output.tif", labels)