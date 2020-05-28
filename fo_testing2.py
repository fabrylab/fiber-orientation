
from fibre_orientation_structure_tensor import *
import pyTFM.plotting
from fo_testing3 import*
#image = "/home/user/Desktop/fibre_orientation_test/pure_grid-0.png"
#image = "/home/user/Desktop/fibre_orientation_test/pure_grid-1.png"
image = "/home/user/Desktop/fibre_orientation_test/2.jpg"
im = gaussian(plt.imread(image),sigma=0.5)

# arr = normalizing(np.mean(im,axis=2))
arr = 1-normalizing(np.mean(im,axis=2))[260:295,120:155]

# arr = np.zeros((100,100))
# arr[50] = 1


## simple example
grady = np.gradient(arr,axis=0)
gradx = np.gradient(arr,axis=1)
plt.figure();plt.imshow(arr, origin="lower"); plt.colorbar()
plt.figure();plt.imshow(grady, cmap="coolwarm", origin="lower"); plt.colorbar()
plt.figure();plt.imshow(gradx, cmap="coolwarm", origin="lower"); plt.colorbar()
show_quiver(-gradx,-grady, filter=[0,1], headwidth=5, cmap="coolwarm",ax_origin="lower")

ori_list, angs = analyze_area_full_orientation(arr, None, points=1000, length=np.pi * 2)
full_angle_plot(ori_list, angs, None, name=None)




# collagen aorund spheroid
im = plt.imread("/home/user/Desktop/fibre_orientation_test/pure_grid-0.png")
sigma1 = 0.1
im_f = gaussian(im, sigma=sigma1)

ori, max_evec, min_evec, max_eval, min_eval = analyze_local(im_f, sigma=15, size=50, filter_type="gaussian")
f = np.nanpercentile(ori, 75)
print("cohernecy threshold=", f)
fig, ax = show_quiver(min_evec[:, :, 0] * ori, min_evec[:, :, 1] * ori, filter=[f, 15],
                      scale_factor=0.1,
                      width=0.003, cbar_str="coherency", cmap="viridis")



ori, max_evec, min_evec, max_eval, min_eval = analyze_local(im_f, sigma=20, size=30, filter_type="gaussian")
f = np.nanpercentile(ori[270:350,750:880], 75)
print("cohernecy threshold=", f)

plt.figure();plt.imshow(im_f[270:350,750:880])
fig, ax = show_quiver(min_evec[270:350,750:880, 0] * ori[270:350,750:880], min_evec[270:350,750:880, 1] * ori[270:350,750:880], filter=[f, 5],
                      scale_factor=0.1,
                      width=0.003, cbar_str="coherency", cmap="viridis", vmax=0.9)





ori, max_evec, min_evec, max_eval, min_eval = analyze_local(im_f, sigma=3, size=None, filter_type="gaussian")
print("cohernecy threshold=", f)

plt.figure();plt.imshow(im_f[270:350,750:880])
fig, ax = show_quiver(min_evec[270:350,750:880, 0] * ori[270:350,750:880], min_evec[270:350,750:880, 1] * ori[270:350,750:880], filter=[f, 4],
                      scale_factor=0.1,
                      width=0.003, cbar_str="coherency", cmap="viridis", vmax=0.9)






