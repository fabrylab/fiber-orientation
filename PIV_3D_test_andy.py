import sys
#sys.path.append(r"U:\Dropbox\software-github\fiber-orientation\fiber-orientation")
from PIV_3D_main import *
from PIV_3D_plotting import *






"""
load stacks
"""
# import glob as glob
# out_folder = r"\\131.188.117.96\biophysDS\dboehringer\Platte_4\Software\3d-openpiv\test"
# # making a 3d array from a stack
# folder1=r"\\131.188.117.96\biophysDS\dboehringer\Platte_4\Measurements_NK_TFM\single-cell-tfm-tom-paper\20170914_A172_rep1\Before"
# #images=[os.path.join(folder1,x) for x in os.listdir(folder1) if "*_ch00.tif" in x]
# images= glob.glob(os.path.join(folder1,"*Pos{}*_ch00.tif".format(str(cell).zfill(3))))
# im_shape = plt.imread(images[0]).shape
# sphere1 = np.zeros((im_shape[0],im_shape[1],len(images)))

# for i,im in enumerate(images):
#     sphere1[:,:,i] = np.mean(plt.imread(im),axis=2)

# folder1=r"\\131.188.117.96\biophysDS\dboehringer\Platte_4\Measurements_NK_TFM\single-cell-tfm-tom-paper\20170914_A172_rep1\After"
# images= glob.glob(os.path.join(folder1,"*Pos{}*_ch00.tif".format(str(cell).zfill(3))))
# im_shape = plt.imread(images[0]).shape
# sphere2 = np.zeros((im_shape[0],im_shape[1],len(images)))

# for i,im in enumerate(images):
#     sphere2[:,:,i] = np.mean(plt.imread(im),axis=2)


"""
test stacks
"""



out_folder = r"\\131.188.117.96\biophysDS\dboehringer\Platte_4\Software\3d-openpiv\test"

# create output folder if it does not exist, print warning otherwise
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

#  test single defo
sphere1 = np.zeros((10,10,10))
sphere1[4,4,1] = 1

sphere2 =  np.zeros((10,10,10))
sphere2[4,3,2] = 1



##3d piv
# #  test cube defo
# sphere1 = np.zeros((10,10,10))
# sphere1[5,5,5:6] = 1

# sphere2 =  np.zeros((10,10,10))
# sphere2[5,5,6:7] = 1



# center = (8, 8, 8)
# size = (16, 16, 16)
# distance = np.linalg.norm(np.subtract(np.indices(size).T,np.asarray(center)), axis=len(center))
# sphere1 = np.ones(size) * (distance<=7)


# center = (8, 8, 8)
# size = (16, 16, 16)
# distance = np.linalg.norm(np.subtract(np.indices(size).T,np.asarray(center)), axis=len(center))
# sphere2 = np.ones(size) * (distance<=5)



# # windowsize for stacks
window_size = (2,2,2)
overlap = (0,0,0)#25   #11
search_area = (5,5,5)



n_rows, n_cols, n_z = get_field_shape3d(sphere1.shape, window_size, overlap)
print("needs %s iterations"%str(n_rows))

u, v, w, sig2noise = extended_search_area_piv3D(sphere1, sphere2, window_size, overlap, search_area, subpixel_method='gaussian',
                              sig2noise_method='peak2peak',
                              width=2,
                              nfftx=None,
                              nffty=None, drift_correction = True)

u, v, w, mask = sig2noise_val(u, v, w=w, sig2noise=sig2noise, threshold=1.3)
u, v, w = replace_outliers(u, v, w, max_iter=1, tol=100 , kernel_size=2, method='disk')

scatter_3D(sig2noise, control="size")
quiver_3D(u, v, w)





"""
visualize results
"""

# # add color
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from itertools import chain
# make grid


# np.save(os.path.join(out_folder,"x.npy"), x)
# np.save(os.path.join(out_folder,"y.npy"), y)
# np.save(os.path.join(out_folder,"z.npy"), z)

#cbound=[0,10]
#filter defos - or use 0 100
#mask_filtered = (np.sqrt(u**2+v**2+w**2)>=np.nanpercentile(np.sqrt(u**2+v**2+w**2),90)) &(np.sqrt(u**2+v**2+w**2)<=np.nanpercentile(np.sqrt(u**2+v**2+w**2),100))
#mask_filtered = deformation>2



#plt.savefig(os.path.join(out_folder,"Displacements.png"))

#plt.close()

#plot_3_D_alpha(sphere1)
#plot_3_D_alpha(sphere2)


#plot_3_D_alpha(s2)