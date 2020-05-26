'''
trying to use the structure tensor to segment zdisks

'''

from fibre_orientation_structure_tensor3 import *

folder = "/home/user/Desktop/ingo_fiber_orientations/"
db = clickpoints.DataFile(os.path.join(folder, "db.cdb"))

out_folder = "/home/user/Desktop/ingo_fiber_orientations/sigma_test2"
createFolder(out_folder)
i = db.getImages()[18]
im = i.data
mask = db.getMask(image=i).data
mask = binary_fill_holes(mask)


im_f = gaussian(im, sigma=0) ### no filter is probabaly best for this purpose
plt.figure()
plt.imshow(im)
grad_y = np.gradient(im, axis=0)  # paramteres: spacing-> set higher dx and dy edge-order: some interpolation (?)
grad_x = np.gradient(im, axis=1)
plt.figure();plt.imshow(grad_x)
plt.figure();plt.imshow(grad_y)
ori, max_evec, min_evec, max_eval, min_eval = analyze_local(im_f, sigma=1, size=40, filter_type="uniform")
plt.figure()
plt.imshow(ori, vmax=1)
plt.colorbar()






f = np.nanpercentile(ori, 75)
print("cohernecy threshold=", f)
fig, ax = show_quiver(min_evec[:, :, 0] * ori, min_evec[:, :, 1] * ori, filter=[f, 12],
                      scale_factor=0.1,
                      width=0.003, cbar_str="coherency", cmap="viridis")

ax.imshow(circ, cmap="spring", vmin=0, vmax=1)