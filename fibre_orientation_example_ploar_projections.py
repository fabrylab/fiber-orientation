import numpy as np
import matplotlib.pyplot as plt
from PIL import Image






# fibres with out spheroid
im = np.asarray(Image.open("/home/user/Desktop/biophysDS/abauer/testing_orientation/Series001_z000_ch00.tif").convert("L"))
polar_array, max_radius, center= polar_coordinate_transform(im, center="image",radius_res=2000, angle_res=2000)
plt.figure();plt.imshow(im)
plt.figure();plt.imshow(polar_array)


# example of a pure grid
im = np.asarray(Image.open("/home/user/Desktop/biophysDS/abauer/testing_orientation/pure_grid.png").convert("L"))
polar_array, max_radius, center= polar_coordinate_transform(im, center="image",radius_res=2000, angle_res=2000)
plt.figure();plt.imshow(im,cmap="Greys_r")
plt.figure();plt.imshow(polar_array)

# slightly shifted image
im = np.asarray(Image.open("/home/user/Desktop/biophysDS/abauer/testing_orientation/pure_grid.png").convert("L"))
polar_array, max_radius, center= polar_coordinate_transform(im, center=(200,200),radius_res=2000, angle_res=2000)
plt.figure();plt.imshow(im,cmap="Greys_r")
plt.figure();plt.imshow(polar_array,cmap="Greys_r")

