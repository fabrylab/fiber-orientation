import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from fibre_orientation import polar_coordinate_transform





# fibres with out spheroid
im = np.asarray(Image.open("evaluation_polar_coordinates/Series001_z000_ch00.tif").convert("L"))
polar_array, max_radius, center, r_factor = polar_coordinate_transform(im, center="image",radius_res=2000, angle_res=2000)
plt.figure();plt.imshow(im)
plt.figure();plt.imshow(polar_array)


# example of a pure grid
im = np.asarray(Image.open("evaluation_polar_coordinates/pure_grid.png").convert("L"))
polar_array, max_radius, center, r_factor = polar_coordinate_transform(im, center="image",radius_res=2000, angle_res=2000)
plt.figure();plt.imshow(im,cmap="Greys_r")
plt.figure();plt.imshow(polar_array)

# slightly shifted image
im = np.asarray(Image.open("evaluation_polar_coordinates/testing_orientation/pure_grid.png").convert("L"))
polar_array, max_radius, center, r_factor = polar_coordinate_transform(im, center=(200,200),radius_res=2000, angle_res=2000)
plt.figure();plt.imshow(im,cmap="Greys_r")
plt.figure();plt.imshow(polar_array,cmap="Greys_r")

