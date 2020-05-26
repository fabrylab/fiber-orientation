'''
"global orientation analysis" --> size filter by applying simple guassian filter
this is most commonly done when they say: "we analyzed the orientation in a region of interesst...

'''
from fibre_orientation_structure_tensor import *
import pyTFM.plotting
from utilities import *
from tqdm import tqdm
import clickpoints
import os
import copy
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.draw import circle






'''
generating clickpoints file

images = []
for fold, subfolder, files in os.walk(folder):
    for f in files:
        if f.endswith(".tif"):
            images.append(os.path.join(fold,f))
db = clickpoints.DataFile(os.path.join(folder,"db.cdb"))
for i in images:
    db.setImage(i)
db.db.close()
'''

def plot1(im, im_f, sigma,  out_folder, ori_res):
    max_evec, min_evec, max_eval, min_eval = ori_res
    circle(r=50, c=im.shape[1] - 50, radius=sigma, shape=im.shape)
    circ = np.zeros(im.shape) + np.nan
    circ[circle(r=50, c=im.shape[1] - 50, radius=sigma, shape=im.shape)] = 1
    grady = np.gradient(im_f, axis=0)
    gradx = np.gradient(im_f, axis=1)
    vmin= np.min(np.stack( [grady,gradx]))
    vmax = np.max(np.stack([grady, gradx]))

    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(im)
    axs[0].imshow(circ, cmap="spring", vmin=0, vmax=1)
    axs[0].arrow(im.shape[1] / 2, im.shape[0] / 2, max_evec[0] * max_eval, max_evec[1] * max_eval, width=10,
                 color="green")
    axs[0].arrow(im.shape[1] / 2, im.shape[0] / 2, min_evec[0] * min_eval, min_evec[1] * min_eval, width=10,
                 color="orange")

    axs[1].imshow(im_f)
    axs[1].set_title("blurred image")
    axs[2].imshow(grady, vmin=vmin,vmax=vmax)
    axs[2].set_title("y gradient")
    im_disp=axs[3].imshow(gradx, vmin=vmin,vmax=vmax)
    axs[3].set_title("x gradient")

    cax= fig.add_axes([0.3,0.1,0.6,0.05])

    plt.colorbar(im_disp, cax=cax, orientation="horizontal")
    fig.savefig(os.path.join(out_folder, "sigma_%s.png" % str(sigma)))



def analyze_local(im, sigma=0, size=0, filter_type="gaussian"):
    if filter_type =="gaussian":
        ot_xx, ot_yx, ot_yy = get_structure_tensor_gaussian(im, sigma)
    if filter_type == "uniform":
        ot_xx, ot_yx, ot_yy = get_structure_tensor_uniform(im, size)

    max_evec, min_evec, max_eval, min_eval = get_principal_vectors(ot_xx, ot_yx, ot_yy)
    ori = (max_eval - min_eval) / (max_eval + min_eval)

    return ori, max_evec, min_evec, max_eval, min_eval





def analyze_local_2_filters(im, sigma1, sigma2=0, window_size=0, filter_type="gaussian"):
     ##### conclusion this should not be necessary: size of the initinaly structure window is fully sufficient for the
     #### selection of  a "structural area"
    ot_xx, ot_yx, ot_yy = get_structure_tensor_gaussian(im, sigma1)
    max_evec, min_evec, max_eval, min_eval = get_principal_vectors(ot_xx, ot_yx, ot_yy)
    ori = (max_eval - min_eval) / (max_eval + min_eval)
    if filter_type == "gaussian":
        ot_xx2 = gaussian(ori ** 2 * min_evec[:, :, 0] * min_evec[:, :, 0], sigma=sigma2)
        ot_yx2 = gaussian(ori ** 2 * min_evec[:, :, 0] * min_evec[:, :, 1],
                         sigma=sigma2)  # ot_yx an dot_xy are mathematically the same
        ot_yy2 = gaussian(ori ** 2 * min_evec[:, :, 1] * min_evec[:, :, 1], sigma=sigma2)


    if filter_type == "uniform":
        ot_xx2 = uniform_filter(ori ** 2 * min_evec[:, :, 0] * min_evec[:, :, 0], size=window_size)
        ot_yx2 = uniform_filter(ori ** 2 * min_evec[:, :, 0] * min_evec[:, :, 1]
                         , size=window_size)  # ot_yx an dot_xy are mathematically the same
        ot_yy2 = uniform_filter(ori ** 2 * min_evec[:, :, 1] * min_evec[:, :, 1], size=window_size)



    max_evec2, min_evec2, max_eval2, min_eval2 = get_principal_vectors(ot_xx2, ot_yx2, ot_yy2)
    ori2 = (max_eval2 - min_eval2) / (max_eval2 + min_eval2)


    return ori2, min_evec2, max_evec2, min_eval2, max_eval2  # min max vectors are exanged here


def analyze_area(im, mask):

    ot_xx, ot_yx, ot_yy = get_structure_tensor_roi(im, mask=mask)
    max_evec, min_evec, max_eval, min_eval = get_principal_vectors(ot_xx, ot_yx, ot_yy)
    ori = (max_eval - min_eval) / (max_eval + min_eval)

    return ori, max_evec, min_evec, max_eval, min_eval

def get_main_orientation_min_squared(ang, vx=0, vy=0):
    ox = np.cos(ang)
    oy = np.sin(ang)
    # scalar product with vector perpendicular to orienationvector needs to be minimized
    ori = np.sum((ox * vx + oy * vy)**2)
    return ori

def analyze_area_full_orientation(im, mask, points=1000, length=np.pi):

    grad_y = np.gradient(im, axis=0)  # paramteres: spacing-> set higher dx and dy edge-order: some interpolation (?)
    grad_x = np.gradient(im, axis=1)

    # orientation tensor
    if not isinstance(mask, np.ndarray):
        mask = np.ones(grad_y.shape).astype(bool)
    else:
        mask = mask.astype(bool)
    oris = []
    angs = np.linspace(0, length, points)
    for ang in angs:
        oris.append(get_main_orientation_min_squared(ang, vx=grad_x[mask], vy=grad_y[mask]))
    oris = np.array(oris)
    return oris, angs

def full_angle_plot(ori_list, angs, out_folder, sigma=""):
    fig = plt.figure()
    ax = plt.subplot(111, projection="polar")
    ax.plot(angs, ori_list)
    fig.savefig(os.path.join(out_folder, "angle_dist_sigma%s.png"%sigma))

if __name__ == "__main__":

    image = "/home/user/Desktop/fibre_orientation_test/5.jpg"
    im = np.mean(plt.imread(image), axis=2)
    im = 1 - im

    ## first try structure tenosr of whole image...
    ot_xx, ot_yx, ot_yy = get_structure_tensor_roi(im)
    max_evec, min_evec, max_eval, min_eval = get_principal_vectors(ot_xx, ot_yx, ot_yy)
    ori = (max_eval - min_eval) / (max_eval + min_eval)

    plt.figure()
    plt.imshow(im)
    plt.arrow(im.shape[1] / 2, im.shape[0] / 2, max_evec[0] * max_eval, max_evec[1] * max_eval, width=10, color="green")
    plt.arrow(im.shape[1] / 2, im.shape[0] / 2, min_evec[0] * min_eval, min_evec[1] * min_eval, width=10,
              color="orange")
    plt.text(0, 0, str(np.round(ori, 2)))

    folder = "/home/user/Desktop/ingo_fiber_orientations/"



    db = clickpoints.DataFile(os.path.join(folder, "db.cdb"))

    out_folder = "/home/user/Desktop/ingo_fiber_orientations/sigma_test1"
    createFolder(out_folder)

    for j, i in enumerate(db.getImages()):
        if not j == 7:
            continue
        im = i.data
        mask = db.getMask(image=i).data
        mask = binary_fill_holes(mask)
        ## first try structure tenosr of whole image...

        # theorie: gausfilter should be close to structure size
        oris = []
        sigmas = [1, 4, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 100, 120, 200]  # [1,4, 10, 20, 30, 50]
        plot = True
        for k in tqdm(sigmas):

            im_f = gaussian(im, sigma=k)
            ori, max_evec, min_evec, max_eval, min_eval = analyze_area(im_f, mask)
            ori_list, angs = analyze_area_full_orientation(im_f, mask, points=100, length=np.pi * 2)

            # ot_xx, ot_yx, ot_yy = get_structure_tensor_roi(im_f, mask=mask)
            # max_evec, min_evec, max_eval, min_eval = get_principal_vectors(ot_xx, ot_yx, ot_yy)
            # ori = (max_eval - min_eval) / (max_eval + min_eval)

            oris.append(ori)
            if plot:
                plot1(im, im_f, k, out_folder,[max_evec, min_evec, max_eval, min_eval])
                ori_list, angs = analyze_area_full_orientation(im_f, mask, points=100, length=np.pi * 2)
                full_angle_plot(ori_list, angs, out_folder, sigma=str(k))

        # fig = plt.figure();plt.plot(sigmas,oris)
        # fig.savefig(os.path.join(out_folder, "sigma_overview.png" ))

        ot_xx, ot_yx, ot_yy = get_structure_tensor_roi(im, mask=mask)
        max_evec, min_evec, max_eval, min_eval = get_principal_vectors(ot_xx, ot_yx, ot_yy)
        ori = (max_eval - min_eval) / (max_eval + min_eval)

        plt.figure()
        plt.imshow(im)
        mask_show = copy.deepcopy(mask).astype(float)
        mask_show[~mask] = np.nan
        plt.imshow(mask_show, alpha=0.4, cmap="Oranges")

        plt.arrow(im.shape[1] / 2, im.shape[0] / 2, max_evec[0] * max_eval, max_evec[1] * max_eval, width=10,
                  color="green")
        plt.arrow(im.shape[1] / 2, im.shape[0] / 2, min_evec[0] * min_eval, min_evec[1] * min_eval, width=10,
                  color="orange")
        plt.text(0, 0, str(np.round(ori, 2)))
