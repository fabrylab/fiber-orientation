import sys
sys.path.append(r"U:\Dropbox\software-github\fiber-orientation\fiber-orientation")
from PIV_3D_main import *




# # windowsize for stacks
window_size = (30,30,30)
overlap =15  #25   #11
search_area = (60,60,60)





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
# sphere1 = np.zeros((10,10,10))
# sphere1[4,4,1] = 1

# sphere2 =  np.zeros((10,10,10))
# sphere2[4,3,2] = 1

#  test cube defo
# sphere1 = np.zeros((10,10,10))
# sphere1[5,5,3:7] = 1

# sphere2 =  np.zeros((10,10,10))
# sphere2[5,5,3:8] = 1


# test sphere defo
center = (50, 50, 50)
size = (100, 100, 100)
distance = np.linalg.norm(np.subtract(np.indices(size).T,np.asarray(center)), axis=len(center))
sphere1 = np.ones(size) * (distance<=30)

center = (50, 50, 50)
size = (100, 100, 100)
distance = np.linalg.norm(np.subtract(np.indices(size).T,np.asarray(center)), axis=len(center))
sphere2 = np.ones(size) * (distance<=15)

#Plot sphere
# fig =plt.figure(figsize=(6,6))
# ax = fig.gca(projection='3d')
# ax.voxels(sphere2, facecolors="r", edgecolor='k',alpha=0.4)
# ax.voxels(sphere1, facecolors="b", edgecolor='k',alpha=0.4)
# plt.show()

#
# ax = fig.gca(projection='3d')
# ax.voxels(sphere2, facecolors="r", edgecolor='k',alpha=0.4)
# ax.voxels(sphere1, facecolors="b", edgecolor='k',alpha=0.4)
# plt.show()


##3d piv
# #  test cube defo
# sphere1 = np.zeros((10,10,10))
# sphere1[5,5,5:6] = 1

# sphere2 =  np.zeros((10,10,10))
# sphere2[5,5,8:9] = 1



# center = (8, 8, 8)
# size = (16, 16, 16)
# distance = np.linalg.norm(np.subtract(np.indices(size).T,np.asarray(center)), axis=len(center))
# sphere1 = np.ones(size) * (distance<=7)


# center = (8, 8, 8)
# size = (16, 16, 16)
# distance = np.linalg.norm(np.subtract(np.indices(size).T,np.asarray(center)), axis=len(center))
# sphere2 = np.ones(size) * (distance<=5)




n_rows, n_cols, n_z = get_field_shape3d(sphere1.shape, window_size, overlap)
print("needs %s iterations"%str(n_rows))

u, v, w, sig2noise = extended_search_area_piv3D(sphere1, sphere2, window_size, overlap, search_area, subpixel_method='gaussian',
                              sig2noise_method='peak2peak',
                              width=2,
                              nfftx=None,
                              nffty=None, drift_correction = True)

# # drift correction
# u= u - np.mean(u)
# v= v - np.mean(v)
# w= w - np.mean(w)


np.save(os.path.join(out_folder,"u.npy"), u)
np.save(os.path.join(out_folder,"v.npy"), v)
np.save(os.path.join(out_folder,"w.npy"), w)
np.save(os.path.join(out_folder,"sig_noise.npy"), sig2noise)

np.savetxt(os.path.join(out_folder,"window_size.txt"), window_size)
np.savetxt(os.path.join(out_folder,"search_size.txt"), search_area)
np.savetxt(os.path.join(out_folder,"overlap.txt"), [overlap])

"""
visualize results
"""

# # add color
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from itertools import chain
# make grid
x, y, z = np.indices(u.shape)
distance = np.sqrt(x**2+y**2+z**2)
deformation = np.sqrt(u**2+v**2+w**2)

# np.save(os.path.join(out_folder,"x.npy"), x)
# np.save(os.path.join(out_folder,"y.npy"), y)
# np.save(os.path.join(out_folder,"z.npy"), z)

#cbound=[0,10]
#filter defos - or use 0 100
#mask_filtered = (np.sqrt(u**2+v**2+w**2)>=np.nanpercentile(np.sqrt(u**2+v**2+w**2),90)) &(np.sqrt(u**2+v**2+w**2)<=np.nanpercentile(np.sqrt(u**2+v**2+w**2),100))
#mask_filtered = deformation>2
mask_filtered = np.ones(x.shape).astype(bool)

offset = np.random.uniform(0,0.2,x[mask_filtered].shape)
xf = x[mask_filtered]
yf = y[mask_filtered]
zf = z[mask_filtered]
uf = u[mask_filtered]
vf = v[mask_filtered]
wf = w[mask_filtered]
df = deformation[mask_filtered]

# make cmap
cbound=[0, np.nanmax(df)]
# create normalized color map for arrows
norm = matplotlib.colors.Normalize(vmin=cbound[0],vmax= cbound[1] ) # 10 ) #cbound[1] ) #)
sm = matplotlib.cm.ScalarMappable(cmap="jet", norm=norm)
sm.set_array([])
# different option
#colors = sm.to_rgba(np.ravel(deformation))
colors = matplotlib.cm.jet(norm(df)) #
# plot the data
#colors[deformation==1.5]=np.array([1,1,1,1])

colors = [c for c,d in zip(colors, df) if d > 0] + list(chain(*[[c, c] for c,d in zip(colors, df) if d > 0]))
# colors in ax.quiver 3d is really fucked up/ will probabaly change with updates:
# requires list with: first len(u) entries define the colors of the shaft, then the next len(u)*2 entries define
# the color of alternating left and right arrow head side. Try for example:
# colors = ["red" for i in range(len(cf))] + list(chain(*[["blue", "yellow"] for i in range(len(cf))]))
# to see this effect
# BUT WAIT THERS MORE: zeor length arrows are apparently filtered out in the matplolib with out filtering the color list appropriately
# so we have to do this our selfes as well



fig = plt.figure()
ax = fig.gca(projection='3d', rasterized=True)

quiver_filtered = ax.quiver(xf, yf, zf, uf, vf, wf ,
                          colors=colors, normalize=True,alpha =0.8, arrow_length_ratio=0,  pivot='tip', linewidth=0.5)
plt.colorbar(sm)
ax.set_xlim(x.min(),x.max())
ax.set_ylim(y.min(),y.max())
ax.set_zlim(z.min(),z.max())
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.w_xaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
ax.w_yaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
ax.w_zaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))


#plt.savefig(os.path.join(out_folder,"Displacements.png"))

#plt.close()

plot_3_D_alpha(sphere1)

plot_3_D_alpha(sphere2)

s2=sig2noise.copy()
s2[sig2noise==1]=0
#plot_3_D_alpha(s2)