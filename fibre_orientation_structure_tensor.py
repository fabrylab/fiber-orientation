### following wiki "structure tensor" /Orientation J

import matplotlib.pyplot as plt
import numpy as np
from fibre_orientation import normalizing
from skimage.filters import gaussian
from pyTFM.plotting import show_quiver
from angel_calculations import project_angle
from scipy.ndimage.filters import uniform_filter


def rotate_vector_field(p, r):
    r_n = np.zeros(r.shape) + np.nan
    if len(r.shape) == 3: # vector field
        r_n[:, :, 0] = + np.cos(p) * (r[:, :, 0]) - np.sin(p) * (r[:, :, 1])  # rotation # by aplying rotation matrix
        r_n[:, :, 1] = + np.sin(p) * (r[:, :, 0]) + np.cos(p) * (r[:, :, 1])

    if len(r.shape) == 1: # single vector
        r_n[0] = + np.cos(p) * (r[0]) - np.sin(p) * (r[1])  # rotation # by aplying rotation matrix
        r_n[1] = + np.sin(p) * (r[0]) + np.cos(p) * (r[1])


    return r_n


def eigen_vec(eval, a, b, d):
    ## do i really not need a??

    # eval must by 2d array
    # eigenvector of symmetric (!) matrix [[a,b][c,d]]

    x = b / np.sqrt(b ** 2 + (eval - a) ** 2)
    y = (eval - a) / np.sqrt(b ** 2 + (eval - a) ** 2)
    return np.stack([x, y], axis=len(y.shape))


def select_max_min(x1, x2, b1, b2):
    # sort x1 and x2 into an array with smaller values
    x1 = np.array(x1)
    x2 = np.array(x2)
    b1 = np.array(b1)
    b2 = np.array(b2)

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


def get_orientation(vx, vy):
    # calculates the angel of a vector with x component vx and y component vy in a range form 0 to np.pi
    # so that angles of paralel vectors pointing in oposite directions are equal

    o = np.arctan2(vy, vx)  # inverse xy!
    # projects angle to range [0,2 pi]
    # equivalent to max_orientation[<0] = 2*np.pi-max_orientation[<0]
    o = o % (2 * np.pi)
    # projecting angles to range of [0,pi]
    o[o > np.pi] = o[o > np.pi] - np.pi
    return o


def produce_orientation_filter1(vx, vy):
    ### very iteresitg result, but not quite what i ant......

    # all paralel vectors in the same direction
    # [-1,-1] gets converted to [1,1]
    # vector length is retained
    o = get_orientation(vx, vy)
    # filtering: big problem: angles such as 0 and np.pi are "almost the same" for our purposes (orientation)
    # strategy: calnulate the diffrence of the orientation to 90 degree line and filter
    # apply spacial filter to this difference reapply the diffrence with corrcet sign
    # amsk that defines if the angle diffrecne is positive or negative (i.e. left or right)
    orient_diff = o - np.pi / 2
    dif_sign = orient_diff > 0
    orient_diff = np.abs(o - np.pi / 2)
    orient_diff = gaussian(orient_diff, sigma=i)
    no = np.zeros(o.shape)
    no[dif_sign] = np.pi / 2 + orient_diff[dif_sign]
    no[~dif_sign] = np.pi / 2 - orient_diff[~dif_sign]

    abs_length = np.sqrt(vx ** 2 + vy ** 2)
    nx = np.cos(no)
    ny = np.sin(no)
    # line paralell to y axis makes problems // must be some better solution
    # nx[np.logical_and(nx==-1,ny==0)] = 1
    nx *= abs_length
    ny *= abs_length

    ## filter the angle diffrence

    return nx, ny



def get_structure_tensor_gaussian(im, sigma):
    # see wikipedia "structure tensor"
    # structure_tenros =[[sum(W * grad_x * grad_ x), sum(W * grad_y * gra_x)],[sum(W * grad_y * grad_x), sum(W * grad_y * grad_Y)}}
    # W: some kind of weighting function, most commonly gaussian, also defines window size for summation// must set sum to 1
    # interpretation: analyzing the (absolute) the gradient in all direction --> high gradient means less coherence in any direction
    #

    grad_y = np.gradient(im, axis=0)  # paramteres: spacing-> set higher dx and dy edge-order: some interpolation (?)
    grad_x = np.gradient(im, axis=1)

    # orientation tensor
    ot_xx = gaussian(grad_x * grad_x, sigma=sigma)
    ot_yx = gaussian(grad_y * grad_x, sigma=sigma)  # ot_yx an dot_xy are mathematically the same
    ot_yy = gaussian(grad_y * grad_y, sigma=sigma)

    return ot_xx, ot_yx, ot_yy





def get_structure_tensor_uniform(im, size):
    # see wikipedia "structure tensor"
    # structure_tenros =[[sum(W * grad_x * grad_ x), sum(W * grad_y * gra_x)],[sum(W * grad_y * grad_x), sum(W * grad_y * grad_Y)}}
    # W: some kind of weighting function, most commonly gaussian, also defines window size for summation// must set sum to 1
    # interpretation: analyzing the (absolute) the gradient in all direction --> high gradient means less coherence in any direction
    #

    # size gives length of edge of filter window

    grad_y = np.gradient(im, axis=0)  # paramteres: spacing-> set higher dx and dy edge-order: some interpolation (?)
    grad_x = np.gradient(im, axis=1)

    # orientation tensor
    ot_xx = uniform_filter(grad_x * grad_x, size=(size,size))
    ot_yx = uniform_filter(grad_y * grad_x, size=(size,size))
    ot_yy = uniform_filter(grad_y * grad_y, size=(size,size))

    return ot_xx, ot_yx, ot_yy


def get_structure_tensor_roi(im, mask=None):
    # structure tensor over specific region of interest with uniform weight

    grad_y = np.gradient(im, axis=0)  # paramteres: spacing-> set higher dx and dy edge-order: some interpolation (?)
    grad_x = np.gradient(im, axis=1)

    # orientation tensor
    if not isinstance(mask, np.ndarray):
        mask = np.ones(grad_y.shape).astype(bool)
    else:
        mask = mask.astype(bool)
    ot_xx = np.mean(grad_x[mask] * grad_x[mask])
    ot_yx = np.mean(grad_y[mask] * grad_x[mask])
    ot_yy = np.mean(grad_y[mask] * grad_y[mask])

    return ot_xx, ot_yx, ot_yy


def get_principal_vectors(ot_xx, ot_yx, ot_yy):
    # --> the same as min max principal stress // from https://www.soest.hawaii.edu/martel/Courses/GG303/Eigenvectors.pdf
    eval1 = (ot_xx + ot_yy) / 2 + np.sqrt(((ot_xx - ot_yy) / 2) ** 2 + ot_yx ** 2)
    eval2 = (ot_xx + ot_yy) / 2 - np.sqrt(((ot_xx - ot_yy) / 2) ** 2 + ot_yx ** 2)

    evec1 = eigen_vec(eval1, ot_xx, ot_yx, ot_yy)
    evec2 = eigen_vec(eval2, ot_xx, ot_yx, ot_yy)

    # we actally want the min eigenvalue and eigen vector
    max_eval, min_eval = select_max_min(eval1, eval2, eval1, eval2)
    max_evec, min_evec = select_max_min(evec1, evec2, eval1, eval2)

    # sometimes minimal vector is not defined, in this case create min eigenvector perpendicular to max eigenvector
    min_not_defined = np.logical_and(np.isnan(min_evec), ~np.isnan(max_evec))
    min_evec[min_not_defined] = rotate_vector_field(np.pi / 2, max_evec)[min_not_defined]
    # fill nans with zeros --> makes sense because later weighting with coherency would set zero  anyway// enabels (gauss) fitlering
    min_evec[np.isnan(min_evec)] = 0
    max_evec[np.isnan(max_evec)] = 0
    return max_evec, min_evec, max_eval, min_eval


if __name__ == "__main__":
    test_vecs = np.zeros((10, 10))

    center = np.array([test_vecs.shape[0], test_vecs.shape[1]]) / 2
    r_vecs = np.meshgrid(np.arange(test_vecs.shape[0]), np.arange(test_vecs.shape[1]), indexing="xy")
    rv_x, rv_y = [r_vecs[1].T - center[1], r_vecs[0].T - center[0]]


    def find_main_oientation(vx, vy):
        # finds the maximum of sum(abs(r*o)), r: vector indicating local orientation (length of this vector will act like
        # no easy maximization posible??

        ox = np.sum(vy) / np.sum(vx)
        oy = np.sqrt(1 - ox ** 2)


    def get_main_orientation(ox, vx, vy):
        oy = np.sqrt(1 - ox ** 2)  # o is vector og length 1
        ori = np.sum(np.abs(ox * vx + oy * vy))
        return ori


    def get_main_orientation_min(ang, vx=0, vy=0):
        ox = np.cos(ang)
        oy = np.sin(ang)

        # scalar product with vector perpendicular to orienationvector needs to be minimized
        ori = np.sum(np.abs(ox * vx + oy * vy))
        return ori


    def get_main_orientation_min_squared(ang, vx=0, vy=0):
        ox = np.cos(ang)
        oy = np.sin(ang)
        # scalar product with vector perpendicular to orienationvector needs to be minimized
        ori = np.sum((ox * vx + oy * vy)**2)
        return ori


    ### there is no single minimum/maximum !!!

    vx = np.random.uniform(-1, 1, (10, 10))
    vy = np.random.uniform(-1, 1, (10, 10))
    vx = np.zeros((10,10)) +1
    vy = np.zeros((10,10)) -1


    im = 1 - np.mean(plt.imread( "/home/user/Desktop/fibre_orientation_test/2.jpg"), axis=2)
    vy = np.gradient(im, axis=0)  # paramteres: spacing-> set higher dx and dy edge-order: some interpolation (?)
    vx = np.gradient(im, axis=1)

    # structure tensor like tensor
    ot_xx = np.sum(vx * vx)
    ot_yx = np.sum(vy * vx)
    ot_yy = np.sum(vy * vy)

    eval1 = (ot_xx + ot_yy) / 2 + np.sqrt(((ot_xx - ot_yy) / 2) ** 2 + ot_yx ** 2)
    eval2 = (ot_xx + ot_yy) / 2 - np.sqrt(((ot_xx - ot_yy) / 2) ** 2 + ot_yx ** 2)

    evec1 = eigen_vec(eval1, ot_xx, ot_yx, ot_yy)
    evec2 = eigen_vec(eval2, ot_xx, ot_yx, ot_yy)
    fig, ax = show_quiver(vx, vy, filter=[0,5])
    ax.arrow(vx.shape[1]/2, vx.shape[0]/2, evec1[0]*vx.shape[0]/4, evec1[1]*vx.shape[0]/4, color="green", head_width=0.2, width=0.1)
    ax.arrow(vx.shape[1]/2, vx.shape[0]/2, evec2[0]*vx.shape[0]/4, evec2[1]*vx.shape[0]/4, color="green", head_width=0.2, width=0.1)

    from scipy.optimize import minimize, basinhopping
    from tqdm import tqdm

    res_x = []
    res_y = []
    for i in tqdm(np.linspace(0, np.pi, 100)):
        res = minimize(get_main_orientation_min, x0=i, args=(vx, vy),
                       method="Nelder-Mead")  # this isused for non linear problems??- guess not
        res_x.append(res["x"])
        res_y.append(res["fun"])
    min_x = res_x[np.argmin(res_y)]
    min_y = res_y[np.argmin(res_y)]

    ra = np.linspace(0, np.pi, 1000)
    ol1 = []
    for i in ra:
        ol1.append(get_main_orientation_min(i, vx, vy))
    ol1 = np.array(ol1)/ np.max(ol1)
    ol2 = []
    for i in ra:
        ol2.append(get_main_orientation_min_squared(i, vx, vy))
    ol2 = np.array(ol2) / np.max(ol2)
    plt.figure();
    plt.plot(ra, ol1)
    plt.plot(ra, ol2, color="orange")


    plt.vlines(min_x, np.min(ol1), np.max(ol1), color="blue")
    plt.vlines(ra[np.argmin(ol2)], np.min(ol1), np.max(ol1), color="orange")
    ea1 = np.arccos(evec1[0]/np.linalg.norm(evec1))
    ea2 = np.arccos(evec2[0]/np.linalg.norm(evec2))
    eamax = ea1 if eval1 > eval2 else ea2
    eamin = eamax + np.pi/2 if eamax < np.pi-np.pi/2 else eamax - np.pi/2

    plt.vlines( eamax, np.min(ol1), np.max(ol1), color="green")
    plt.vlines( eamin, np.min(ol1), np.max(ol1), color="green")









    f, a = show_quiver(rv_x, rv_y, scale_ratio=0.1, filter=[0, 1], alpha=0)
    im = a.imshow(get_orientation(rv_x, rv_y),
                  alpha=1)  # cmap="hsv""cyclic colrmap for representation is very importatn
    plt.colorbar(im)

    # f, a = show_quiver(rv_re_x, rv_re_y, scale_ratio=0.1, filter=[0, 1], alpha=0)
    # im = a.imshow(get_orientation(rv_re_x, rv_re_y),
    #              alpha=1)  # cmap="hsv""cyclic colrmap for representation is very importatn
    # plt.colorbar(im)

    image = "/home/user/Desktop/fibre_orientation_test/pure_grid-0.png"
    # image = "/home/user/Desktop/fibre_orientation_test/pure_grid-1.png"
    # image = "/home/user/Desktop/fibre_orientation_test/6.jpg"
    im = plt.imread(image)
    arr = im
    # arr = normalizing(np.mean(im,axis=2))
    # arr = 1-normalizing(np.mean(im,axis=2))

    # arr = np.zeros((100,100))
    # arr[50] = 1

    # structure_tenros =[[sum(W * grad_x * grad_ x), sum(W * grad_y * gra_x)],[sum(W * grad_y * grad_x), sum(W * grad_y * grad_Y)}}
    # W: some kind of weighting function, most commonly gaussian, also defines window size for summation// must set sum to 1
    # interpretation: analyzing the (absolute) the gradient in all direction --> high gradient means less coherence in any direction
    #

    ot_xx, ot_yx, ot_yy = get_structure_tensor(arr, sigma=4)
    max_evec, min_evec, max_eval, min_eval = get_principal_vectors(ot_xx, ot_yx, ot_yy)
    proj_orientation = get_orientation(min_evec[:, :, 0], min_evec[:, :, 1])

    coherency = (np.abs(max_eval) - np.abs(min_eval)) / (np.abs(max_eval) + np.abs(min_eval))

# todo: fix orietnation of vectors--> project angles//
# todo: play aorunf with sigma
# todo: think about weightin with image intensity and or thresholding/filtering of raw image
