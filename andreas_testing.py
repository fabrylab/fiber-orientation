import sys
from openpiv.pyprocess3D import extended_search_area_piv3D as esa_3d
# append your folder with PIV_3D_main and plotting here
# sys.path.append(r"U:\Dropbox\software-github\fiber-orientation\fiber-orientation")
sys.path.append(r"U:\Dropbox\software-github\nk-singlecell-3D-TFM")
from PIV_3D_main import extended_search_area_piv3D as esa_3d_fo
from PIV_3D_plotting import quiver_3D as quiver_3D_fo
from PIV_3D_main import sig2noise_val, replace_outliers
from PIV_3D_plotting import animate_stacks
from openpiv.PIV_3D_plotting import quiver_3D as quiver_3D_openpiv
import glob as glob
from tqdm import tqdm
from pyTFM.plotting import show_quiver
from pyTFM.utilities_TFM import createFolder
from openpiv.pyprocess import extended_search_area_piv as esa_2d
from openpiv.pyprocess3D import extended_search_area_piv3D as esa_3d_openpiv
# Full 3D Deformation analysis
import numpy as np
import os
import matplotlib.pyplot as plt

# stack properties
# voxelsize
du = 0.2407
dv = 0.2407
dw = 1.0071
# total image dimesnion for x y z
image_dim = (123.02, 123.02, 122.86)

# times to evaluate - or use list with certain  timesteps [23]
times = np.arange(1, 23)

# keep these values for our nk cells stacks
win_um = 12
fac_overlap = 0.4
signoise_filter = 1.3
# windowsize for stacks
window_size = (int(win_um / du), int(win_um / dv), int(win_um / dw))
overlap = (int(fac_overlap * win_um / du), int(fac_overlap * win_um / dv), int(fac_overlap * win_um / dw))
search_area = (int(win_um / du), int(win_um / dv), int(win_um / dw))

# specify output folder and image stacks within the loop !

# loop through time
times = [0,7]
out_put = createFolder("out")
for t in tqdm(times):
    print(t)
    # create output folder
    out_folder = r"pos02_ref21/t{}".format(str(t))
    # create output folder if it does not exist, print warning otherwise
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    """
    load alive stacks
    """
    folder = r"/home/user/Desktop/biophysDS/dboehringer/Platte_4/Measurements_NK_TFM/18.05.20-NK-TFM/Sample1_MarkandFind_001/Mark_and_Find_001"
    images = glob.glob(
        os.path.join(folder, "Mark_and_Find_001_Pos002_S001_t{}_z*_RAW_ch00.tif".format(str(t).zfill(2))))[30:-30]
    im_shape = plt.imread(images[0]).shape
    alive = np.zeros((im_shape[0], im_shape[1], len(images)))
    for i, im in enumerate(images):
        alive[:, :, i] = plt.imread(im)

    """
    load relaxed stack 
    """
    folder = r"/home/user/Desktop/biophysDS/dboehringer/Platte_4/Measurements_NK_TFM/18.05.20-NK-TFM/Sample1_MarkandFind_001/Mark_and_Find_001"
    images = glob.glob(
        os.path.join(folder, "Mark_and_Find_001_Pos002_S001_t{}_z*_RAW_ch00.tif".format(str(21).zfill(2))))[30:-30]
    im_shape = plt.imread(images[0]).shape
    relax = np.zeros((im_shape[0], im_shape[1], len(images)))
    for i, im in enumerate(images):
        relax[:, :, i] = plt.imread(im)



    # 2d deformation field of projections
    if t == 0:
        s1 = 0
        s2 = 1
    if t == 7:
        s1 = 0
        s2 = 40

    # projection along z-axis
    u2d, v2d = esa_2d(np.max(relax[:, :, s1:-s2], axis=2), np.max(alive[:, :, s1:-s2], axis=2), window_size=60,
                      overlap=30, search_area_size=60)
    u2d -= u2d.mean()
    v2d -= v2d.mean()
    fig, ax = show_quiver(u2d, -v2d)
    fig.savefig(os.path.join(out_put,"def_max_z_proj_t%s.svg"%str(t)))
    ani1, ims = animate_stacks(relax[:, :, s1:-s2], alive[:, :, s1:-s2], interval=100, repeat_delay=0, z_range=[0, 20],
                               gif_name=os.path.join(out_put,"z_proj_t%s.gif" % str(t)))
    # plt.show() # only shows up with plt.show()

    #  projection along x-axis
    ani1, ims = animate_stacks(relax[:, 200:300, :], alive[:, 200:300, :],  interval=100, repeat_delay=0,
                               gif_name=os.path.join(out_put,"y_proj_t%s.gif" % str(t)),max_axis=1, drift_correction=True)
    u2d, v2d, sig2noise = esa_2d(np.max(relax[:, 200:300, :], axis=1), np.max(alive[:, 200:300, :], axis=1), window_size=30,
                      overlap=20, search_area_size=30,sig2noise_method="peak2peak")
    u2d -= u2d.mean()
    v2d -= v2d.mean()
    u2d, v2d,  mask = sig2noise_val(u2d, v2d, sig2noise=sig2noise, threshold=signoise_filter)
    u2d, v2d = replace_outliers(u2d, v2d,max_iter=1, tol=100, kernel_size=2, method='disk')
    fig, ax = show_quiver(u2d, -v2d, width=0.02)
    fig.savefig(os.path.join(out_put,"def_max_x_proj_t%s.svg"%str(t)))


    ##### with local fibre-orientation functions
    u1, v1, w1, sig2noise = esa_3d_fo(relax, alive, window_size, overlap, search_area, du=du, dv=dv,
                                                    dw=dw, subpixel_method='gaussian',
                                                    sig2noise_method='peak2peak',
                                                    width=2,
                                                    nfftx=None,
                                                    nffty=None, drift_correction=True)#
    u1 -= np.nanmean(u1)
    v1 -= np.nanmean(v1)
    w1 -= np.nanmean(w1)
    # filter data
    uf1, vf1, wf1, mask = sig2noise_val(u1, v1, w=w1, sig2noise=sig2noise, threshold=signoise_filter)
    uf1, vf1, wf1 = replace_outliers(uf1, vf1, wf1, max_iter=1, tol=100, kernel_size=2, method='disk')

    # plot 3d quiver
    fig = quiver_3D_fo(uf1, vf1, wf1, image_dim=image_dim, quiv_args={"linewidth": 1, "alpha": 0.3, "length": 2})
    fig.savefig(os.path.join(out_put,"3d_quiv_fo%s.svg"%str(t)))
    # project the deformations
    u_proj1 = np.mean(uf1, axis=2)
    v_proj1 = np.mean(vf1, axis=2)
    fig, ax = show_quiver(u_proj1, v_proj1)
    fig.savefig(os.path.join(out_put,"def_pro_fo%s.svg"%str(t)))

    ##### with local openpiv functions

    ##### with local fibre-orientation functions
    u, v, w, sig2noise = esa_3d_openpiv(relax, alive, window_size=window_size, overlap=overlap, search_area_size=search_area, subpixel_method='gaussian',
                                   sig2noise_method='peak2peak',
                                   width=2,
                                   nfftx=None,
                                   nffty=None)  #
    u -= np.nanmean(u)
    v -= np.nanmean(v)
    w -= np.nanmean(w)

    u *= du
    v *= dv
    w *= dw
    # filter data
    uf, vf, wf, mask = sig2noise_val(u, v, w=w, sig2noise=sig2noise, threshold=signoise_filter)
    uf, vf, wf = replace_outliers(uf, vf, wf, max_iter=1, tol=100, kernel_size=2, method='disk')

        # plot 3d quiver
    fig, ax = quiver_3D_openpiv(uf, -vf, -wf, quiv_args={"linewidth": 1, "alpha": 0.3, "length": 2})
    fig.savefig(os.path.join(out_put,"3d_quiv_openpiv%s.svg" % str(t)))

    # project the deformations
    u_proj = np.mean(-uf, axis=2)
    v_proj = np.mean(vf, axis=2)
    fig, ax = show_quiver(u_proj, v_proj)
    fig.savefig(os.path.join(out_put,"def_pro_openpiv%s.svg" % str(t)))