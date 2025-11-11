from cpptiff import read_tiff, write_tiff
import time
import numpy as np
import pandas as pd
from skimage.filters.rank import median
from skimage.segmentation import watershed
from skimage.morphology import ball, binary_erosion
from skimage.filters import threshold_otsu, rank
import edt
read_im_time_start = time.time()

polyT_im = read_tiff("Dark.tif", [0,300])
segmentation_data = pd.read_csv("DSR_20250704_Pos9_8_1_589_DAPI_item_0005_TwangSegmentation_RegionProps.csv")
seed_labels = segmentation_data["id"].to_numpy()
seed_z = segmentation_data["zpos"].to_numpy()
seed_y = segmentation_data["ypos"].to_numpy()
seed_x = segmentation_data["xpos"].to_numpy()
seed_zyx = np.vstack((seed_z, seed_y, seed_x)).T

thresh = threshold_otsu(polyT_im)
polyT_im_mask = np.array(polyT_im > thresh, dtype=np.uint8)


r = 3
selem = ball(r)

polyT_im_mask = median(polyT_im_mask, selem)
print("Completed median filtering")

dt_time_start = time.time()
dt = edt.edt(
  polyT_im_mask
)
dt_time_end = time.time()
print(f"Took {dt_time_end - dt_time_start} seconds to calculate distance transform.")


# dt = distance_transform_cdt(im_mask)
# dt = np.array(dt, dtype=np.uint8)

write_tiff("../data/images/watershed/DSR_20250704_Pos9_8_2_590_polyT_Dark_0_300_DT.tif", dt)
write_tiff("../data/images/watershed/DSR_20250704_Pos9_8_2_590_polyT_Dark_0_300_mask.tif", polyT_im_mask)



