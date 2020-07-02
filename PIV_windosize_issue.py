# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:openpiv] *
#     language: python
#     name: conda-env-openpiv-py
# ---

# %% [markdown]
# ## OpenPIV tutorial 2
#
# In this notebook we compare the time to run the same analysis using Cython (precompiled) version
# with the Python process using FFT and/or direct cross-correlation method

# %%
from openpiv import tools, process, scaling, pyprocess, validation, filters
import numpy as np
import pylab

# %matplotlib inline

# %%
frame_a  = tools.imread( '/home/andy/Software/openpiv-python/openpiv/examples/test1/exp1_001_a.bmp' )
frame_b  = tools.imread( '/home/andy/Software/openpiv-python/openpiv/examples/test1/exp1_001_b.bmp' )
#frame_a  = tools.imread( '../test1/exp1_001_a.bmp' )
#frame_b  = tools.imread( '../test1/exp1_001_b.bmp' )
#pylab.imshow(np.c_[frame_a,np.ones((frame_a.shape[0],20)),frame_b],cmap=pylab.cm.gray)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
def update_plot(i, ims, ax):
    a1 = ax.imshow(ims[i])
    return [a1]


ims = [frame_a,frame_b]
fig = plt.figure()
ax = plt.gca()
ani = animation.FuncAnimation(fig, update_plot, 2, interval=300, blit=True, repeat_delay=0, fargs=(ims, ax))

# %%
# %%time
u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=24, overlap=12, dt=0.02, search_area_size=64, sig2noise_method='peak2peak' )
#u, v, sig2noise = process.extended_search_area_piv( frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=24, overlap=12, dt=0.02, search_area_size=64, sig2noise_method='peak2peak' )
x, y = process.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )
u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 2.5 )
u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
tools.save(x, y, u, v, mask, 'exp1_001_extended.txt' )

# %%
# %%time
u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, corr_method='fft', window_size=24, overlap=12, search_area_size=24, dt=0.02, sig2noise_method='peak2peak' )
x, y = pyprocess.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )
u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 2.5 )
u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
tools.save(x, y, u, v, mask, 'exp1_001_fft.txt' )

window_size3 = 24
search_area_size3 = 24
u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, corr_method='fft', window_size=window_size3, overlap=12, search_area_size=search_area_size3, dt=0.02, sig2noise_method='peak2peak' )
x, y = pyprocess.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12)
tools.save(x, y, u, v, np.ones(u.shape), 'exp1_001_non_extended.txt' )



fig,ax = tools.display_vector_field('exp1_001_extended.txt', scale=30, width=0.0025)
ax.set_title(" window_size = %s, search_area_size = %s with\nprocess.extended_search_area_piv"%(str(window_size1), str(search_area_size1)))
fig,ax = tools.display_vector_field('exp1_001_extended_py.txt', scale=30, width=0.0025)
ax.set_title(" window_size = %s, search_area_size = %s with\npyprocess.extended_search_area_piv"%(str(window_size2), str(search_area_size2)))
fig,ax = tools.display_vector_field('exp1_001_non_extended.txt', scale=30, width=0.0025)
ax.set_title(" window_size = %s, search_area_size = %s with\npyprocess.extended_search_area_piv"%(str(window_size3), str(search_area_size3)))
