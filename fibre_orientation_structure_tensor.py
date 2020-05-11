### following wiki "structure tensor" /Orientation J

import matplotlib.pyplot as plt
import numpy as np
from fibre_orientation import normalizing
from skimage.filters import gaussian
from pyTFM.plotting import show_quiver

def rotate_vector_field(p, r):

    r_n = np.zeros(r.shape) + np.nan
    r_n[:, :, 0] = + np.cos(p) * (r[:, :, 0]) - np.sin(p) * (r[:, :, 1])  # rotation # by aplying rotation matrix
    r_n[:, :, 1] = + np.sin(p) * (r[:, :, 0]) + np.cos(p) * (r[:, :, 1])

    return r_n



def eigen_vec(eval,a,b,d):
    ## do i really not need a??

    # eval must by 2d array
    # eigenvector of symmetric (!) matrix [[a,b][c,d]]

    x = b / np.sqrt(b ** 2 + (eval - a) ** 2)
    y =  (eval - a) / np.sqrt(b ** 2 + (eval - a) ** 2)
    return np.stack([x,y], axis=2)

def select_max_min(x1, x2, b1, b2):
    # sort x1 and x2 into an array with smaller values
    x_max = np.zeros(x1.shape)
    x_min = np.zeros(x2.shape)

    bigger1 = np.abs(b1) > np.abs(b2)  # mask where absolute value of first eigenvalue is the bigger
    bigger2 = ~bigger1

    x_max[bigger1] = x1[bigger1]
    x_max[bigger2] = x2[bigger2]

    x_min[bigger2] = x1[bigger2]
    x_min[bigger1] = x2[bigger1]

    return x_max, x_min


def gaussian_nan(arr, sigma):
    ## based on https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    ## nmight by probelmatic??

    ##--> check out this weird effet arr=np.zeros((100,100)) +np.nan
    # arr[50] =1
    # arr[49]=0.5
    # plt.figure();plt.imshow(arr),plt.colorbar()
    # plt.figure();plt.imshow(gaussian_nan(arr, 1)),plt.colorbar()

    V = arr.copy()
    V[np.isnan(arr)] = 0
    VV = gaussian(V, sigma=sigma)

    W = 0 * arr.copy() + 1
    W[np.isnan(arr)] = 0
    WW = gaussian(W, sigma=sigma)

    return VV / WW



image = "/home/user/Desktop/fibre_orientation_test/pure_grid-0.png"
#image = "/home/user/Desktop/fibre_orientation_test/pure_grid-1.png"
#image = "/home/user/Desktop/fibre_orientation_test/6.jpg"
im = plt.imread(image)
arr = im
#arr = normalizing(np.mean(im,axis=2))
#arr = 1-normalizing(np.mean(im,axis=2))

#arr = np.zeros((100,100))
#arr[50] = 1

# structure_tenros =[[sum(W * grad_x * grad_ x), sum(W * grad_y * gra_x)],[sum(W * grad_y * grad_x), sum(W * grad_y * grad_Y)}}
# W: some kind of weighting function, most commonly gaussian, also defines window size for summation// must set sum to 1
# interpretation: analyzing the (absolute) the gradient in all direction --> high gradient means less coherence in any direction
#
grad_y = np.gradient(arr, axis=0) # paramteres: spacing-> set higher dx and dy edge-order: some interpolation (?)
grad_x = np.gradient(arr, axis=1)


sigma = 4  # most important parameter (?) defines windowsize and shape of weighting function
# orientation tensor
ot_xx = gaussian(grad_x * grad_x, sigma=sigma)
ot_yx = gaussian(grad_y * grad_x, sigma=sigma) # ot_yx an dot_xy are mathematically the same
ot_yy = gaussian(grad_y * grad_y, sigma=sigma)


eval1 = (ot_xx + ot_yy) / 2 + np.sqrt(((ot_xx - ot_yy) / 2) ** 2 + ot_yx ** 2)#--> the same as min max principal stress // from https://www.soest.hawaii.edu/martel/Courses/GG303/Eigenvectors.pdf
eval2 = (ot_xx + ot_yy) / 2 - np.sqrt(((ot_xx - ot_yy) / 2) ** 2 + ot_yx ** 2)

evec1 = eigen_vec(eval1, ot_xx, ot_yx, ot_yy)
evec2 = eigen_vec(eval2, ot_xx, ot_yx, ot_yy)




# we actally want the min eigenvalue and eigen vector
max_eval, min_eval = select_max_min(eval1, eval2, eval1, eval2)
max_evec, min_evec = select_max_min(evec1, evec2, eval1, eval2)

# sometimes minimal vector is not defined, in this case create min eigenvector perpendicular to max eigenvector
min_not_defined = np.logical_and(np.isnan(min_evec), ~np.isnan(max_evec))
min_evec[min_not_defined] = rotate_vector_field(np.pi/2, max_evec)[min_not_defined]
# fill nans with zeros --> makes sense because later weighting with coherency would set zero  anyway// enabels (gauss) fitlering
min_evec[np.isnan(min_evec)] = 0
max_evec[np.isnan(max_evec)] = 0

coherency = (np.abs(max_eval) - np.abs(min_eval)) / (np.abs(max_eval) + np.abs(min_eval))

# needs to project to range of 0 to pi/2
max_orientation = np.arctan2(max_evec[:,:,0], max_evec[:,:,1])## how to /if to deal with flipped vectors??
min_orientation = np.arctan2(min_evec[:,:,0], min_evec[:,:,1]) ## should always be 90 degree apart
energy = ot_xx + ot_yy  ### what is this and hy do I care?




show_quiver(min_evec[:,:,0] * coherency, min_evec[:,:,1] * coherency, scale_ratio=0.1, filter=[0,14])


plt.figure();plt.imshow(energy); plt.colorbar()
plt.figure();plt.imshow(coherency); plt.colorbar()

min_evec_f =  np.zeros(min_evec.shape)
min_evec_f[:,:,0] = gaussian(min_evec[:,:,0], sigma=10) # filtering might be bad because oppsoing arrows oul cancle each other out and that succs
min_evec_f[:,:,1] = gaussian(min_evec[:,:,1], sigma=10)
show_quiver(min_evec_f[:,:,0] * coherency, min_evec_f[:,:,1] * coherency, scale_ratio=0.1, filter=[0,14])
#show_quiver(max_evec[:,:,0], max_evec[:,:,1], scale_ratio=0.3)
plt.figure();plt.imshow(arr); plt.colorbar()
plt.close("all")


# todo: fix orietnation of vectors--> project angles//
# todo: play aorunf with sigma
# todo: think about weightin with image intensity and or thresholding/filtering of raw image


