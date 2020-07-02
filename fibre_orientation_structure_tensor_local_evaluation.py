# using the structure tensor to locally quantify orientation

import sys
sys.path.append("/home/user/Software/fiber-orientation/")
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from scipy.ndimage.filters import uniform_filter
import os
from fibre_orientation_structure_tensor import*

import numpy as np
from pyTFM.plotting import show_quiver
from skimage.draw import circle
from scipy.signal import convolve2d




# spatial distribution for fibre orientation
# requires pyTFM
import pyTFM.plotting
im = plt.imread("/home/user/Desktop/fibre_orientation_test/pure_grid-0.png")

# image preprocessing// for now just very small blurr, could also use other steps:
# maybe contrast spreading/ local contrast spreading....
sigma1 = 1
im_f = gaussian(im, sigma=sigma1)
# claculating local orientations
ori, max_evec, min_evec, max_eval, min_eval = analyze_local(im_f, sigma=15, size=50, filter_type="gaussian")
# finding nice threshold for orientation display
f = np.percentile(ori, 75)
# plotting orientation
# filter controls [minimum value("arrow lenght) displayed, only every x-th arrow shown]
fig, ax =  show_quiver(min_evec[:, :, 0] * ori, min_evec[:, :, 1] * ori, filter=[f, 15],
                      scale_factor=0.1,
                      width=0.003, cbar_str="coherency", cmap="viridis")

plt.figure();plt.imshow(im)
