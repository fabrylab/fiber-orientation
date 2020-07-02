from openpiv import tools, process, scaling, pyprocess, validation, filters
import numpy as np
import pylab
# %matplotlib inline

# %%
frame_a  = tools.imread( '../openpiv-python/openpiv/examples/test1/exp1_001_a.bmp' )
frame_b  = tools.imread( '../openpiv-python/openpiv/examples/test1/exp1_001_b.bmp' )
#pylab.imshow(np.c_[frame_a,np.ones((frame_a.shape[0],20)),frame_b],cmap=pylab.cm.gray)


# %%



shape1=np.zeros((20,20))
shape2=np.zeros((20,20))
shape1[10,9:14] = 1
shape2[10,10:15] = 1
for sa in [4,5,6,7]:
    u1, v1, sig2noise = pyprocess.extended_search_area_piv(shape1.astype(np.int32), shape2.astype(np.int32), corr_method='fft',
                                                           window_size=4, overlap=2, dt=0.02, search_area_size=sa,
                                                           sig2noise_method='peak2peak')

    x1, y1 = pyprocess.get_coordinates(image_size=shape2.shape, window_size=4, overlap=2)

    tools.save(x1, y1, u1, v1, np.ones(u1.shape), 'test1.txt')

    tools.display_vector_field('test1.txt', scale=50, width=0.0025, scale_units="inches")

for sa in [4,5,6,7]:
    u1, v1, sig2noise = process.extended_search_area_piv(shape1.astype(np.int32), shape2.astype(np.int32),
                                                           window_size=4, overlap=2, dt=0.02, search_area_size=sa,
                                                           sig2noise_method='peak2peak')

    x1, y1 = pyprocess.get_coordinates(image_size=shape2.shape, window_size=4, overlap=2)

    tools.save(x1, y1, u1, v1,  np.ones(u1.shape), 'test1.txt')

    tools.display_vector_field('test1.txt', scale=50, width=0.0025, scale_units="inches")



