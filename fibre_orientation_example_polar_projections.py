import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.append("/home/user/Software/fiber-orientation")
from fibre_orientation import polar_coordinate_transform

import os
os.getcwd()


# fibres with out spheroid
im = np.asarray(Image.open(r"evaluation_polar_coordinates/Pos002_S001_t314_z6_ch00.tif").convert("L"))
polar_array, max_radius, center, r_factor = polar_coordinate_transform(im, center="image",radius_res=2000, angle_res=2000)
plt.figure();plt.imshow(im)
plt.figure();plt.imshow(polar_array)
plt.show()

# example of a pure grid
im = np.asarray(Image.open(r"evaluation_polar_coordinates/pure_grid.png").convert("L"))
polar_array, max_radius, center, r_factor = polar_coordinate_transform(im, center="image",radius_res=2000, angle_res=2000)
plt.figure();plt.imshow(im,cmap="Greys_r")
plt.figure();plt.imshow(polar_array)
plt.show()
# slightly shifted image
im = np.asarray(Image.open(r"evaluation_polar_coordinates/pure_grid.png").convert("L"))
polar_array, max_radius, center, r_factor = polar_coordinate_transform(im, center=(200,200),radius_res=2000, angle_res=2000)
plt.figure();plt.imshow(im,cmap="Greys_r")
plt.figure();plt.imshow(polar_array,cmap="Greys_r")
plt.show()
