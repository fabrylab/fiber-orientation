
from fibre_orientation_structure_tensor import *
import pyTFM.plotting

#image = "/home/user/Desktop/fibre_orientation_test/pure_grid-0.png"
#image = "/home/user/Desktop/fibre_orientation_test/pure_grid-1.png"
image = "/home/user/Desktop/fibre_orientation_test/5.jpg"
im = plt.imread(image)
arr = im
# arr = normalizing(np.mean(im,axis=2))
arr = 1-normalizing(np.mean(im,axis=2))[100:170,201:280]

# arr = np.zeros((100,100))
# arr[50] = 1



ot_xx, ot_yx, ot_yy = get_structure_tensor_gaussian(arr, sigma=2)
max_evec, min_evec, max_eval, min_eval = get_principal_vectors(ot_xx, ot_yx, ot_yy)
proj_orientation = get_orientation(min_evec[:, :, 0], min_evec[:, :, 1])

coherency = (np.abs(max_eval) - np.abs(min_eval)) / (np.abs(max_eval) + np.abs(min_eval))


coherency[coherency<0.6]=0 # filtering mainly for displaying!!!
vecw = min_evec * coherency[:,:,None] * arr[:,:,None] # weighting with pixel intenisty// maybe use some genral thresholding
show_filter=np.percentile(np.linalg.norm(vecw, axis=2),80)

show_quiver(vecw[:,:,0],vecw[:,:,1], filter=[show_filter, 8], scale_factor=0.1, width=0.003,  cbar_str="weighted coherency")
plt.figure();plt.imshow(arr)
plt.figure();plt.imshow(coherency);
plt.colorbar().set_label("coherency")


