from openpiv.process import extended_search_area_piv as e_cython
from openpiv.pyprocess import extended_search_area_piv as e_python
import numpy as np
import matplotlib.pyplot as plt
import time
from PIV_3D_2 import *
from PIV_3D_plotting import *
from PIV_3D_main import extended_search_area_piv3D as piv3D_old
from scipy import correlate
from openpiv.pyprocess import extended_search_area_piv
from pyTFM.plotting import show_quiver
'''
test if cython implementation of open piv is faster... answer: it is not

im1 = "/home/user/Desktop/backup_from_harddrive/data_traction_force_microscopy/WT_vs_KO_images/WTshift/01before_shift.tif"
im2 =  "/home/user/Desktop/backup_from_harddrive/data_traction_force_microscopy/WT_vs_KO_images/WTshift/01after_shift.tif"

im1  = plt.imread(im1)[:,:,0].astype("int32")
im2 = plt.imread(im2)[:,:,0].astype("int32")


t1 = time.time()
u, v, sig2noise = e_cython(im1, im2, window_size=100, overlap=50, search_area_size=100, subpixel_method='gaussian',
                              sig2noise_method='peak2peak',
                              width=2,
                              nfftx=None,
                              nffty=None)
t2 = time.time()
print(t2-t1)

t1 = time.time()
u, v, sig2noise = e_python(im1, im2, window_size=100, overlap=30, search_area_size=100, subpixel_method='gaussian',
                              sig2noise_method='peak2peak',
                              width=2,
                              nfftx=None,
                              nffty=None)
t2 = time.time()
print(t2-t1)
'''



window_size = (5,5,5)
overlap = (4,4,4) #25   #11
search_area  = (6,6,6)

center = (8, 8, 8)
size = (16, 16, 16)
distance = np.linalg.norm(np.subtract(np.indices(size).T,np.asarray(center)), axis=len(center))
sphere1 = np.ones(size) * (distance<=7)


center = (8, 8, 8)
size = (16, 16, 16)
distance = np.linalg.norm(np.subtract(np.indices(size).T,np.asarray(center)), axis=len(center))
sphere2 = np.ones(size) * (distance<=5)


sphere1 = np.zeros(size)
sphere2 = np.zeros(size)
sphere1[5,5,3:10] = 1
sphere2[5,5,4:11] = 1


t1 = time.time()
u1, v1, w1, sig2noise1 = extended_search_area_piv3D(sphere1, sphere2, window_size=window_size, overlap=overlap, search_area_size=search_area, subpixel_method='gaussian',
                              sig2noise_method='peak2peak',corr_method="fft",
                              width=2,
                              nfftx=None,
                              nffty=None)


#u, v, w = replace_outliers(u, v, w, max_iter=1, tol=100 , kernel_size=2, method='disk')
t2 = time.time()
print(t2-t1)

#quiver_3D(u, v, w)
t1 = time.time()
u2, v2, w2, sig2noise2 = piv3D_old(sphere1, sphere2, window_size=window_size, overlap=overlap, search_area_size=search_area, subpixel_method='gaussian',
                              sig2noise_method='peak2peak',
                              width=2,
                              nfftx=None,
                              nffty=None)


#u, v, w = replace_outliers(u, v, w, max_iter=1, tol=100 , kernel_size=2, method='disk')
t2 = time.time()
print(t2-t1)





u1, v1, w1, mask = sig2noise_val(u1, v1, w=w1, sig2noise=sig2noise1, threshold=0)
u2, v2, w2, mask = sig2noise_val(u2, v2, w=w2, sig2noise=sig2noise2, threshold=0)

#
#u, v, w = replace_outliers(u, v, w, max_iter=1, tol=100 , kernel_size=2, method='disk')

plt.close("all")
#scatter_3D(sphere1, control="size")
#scatter_3D(sphere2, control="size")
scatter_3D(sig2noise1, control="color")
scatter_3D(sig2noise2, control="color")
quiver_3D(-u1, v1, w1)
quiver_3D(u2, v2, w2)
#u1, v1, w1, mask = sig2noise_val(u1, v1, w=w1, sig2noise=sig2noise, threshold=1.01)
#u, v, w, mask = sig2noise_val(u, v, w=w, sig2noise=sig2noise, threshold=1.3)
#u, v, w = replace_outliers(u, v, w, max_iter=1, tol=100 , kernel_size=2, method='disk')



