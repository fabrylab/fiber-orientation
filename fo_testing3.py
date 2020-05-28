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

from pyTFM.graph_theory_for_cell_boundaries import mask_to_graph, find_path_circular
from skimage.morphology import binary_erosion
from scipy.signal import convolve2d



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




if __name__ == "__main__":
    '''
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
    '''

    folder = "/home/user/Desktop/ingo_fiber_orientations/"
    db = clickpoints.DataFile(os.path.join(folder, "db.cdb"))

    out_folder =  "/home/user/Desktop/ingo_fiber_orientations/analysis1"
    createFolder(out_folder)
    oris = []
    sigma = 5
    nomr1 = 5
    norm2 = 95
    plot = True
    sigma_test = False
    res=[]
    for j, i in tqdm(enumerate(db.getImages())):

        im = i.data
        im_name = i.filename.split(".")[0] + "_" + i.filename.split(" ")[2][:-4]
        print(im_name)
        #if not im_name == "MAX_7500_05022020_Series019":
        #    continue
        mask = db.getMask(image=i).data
        mask = binary_fill_holes(mask)
        ## first try structure tenosr of whole image...

        # theorie: gausfilter should be close to structure size
        im_n = normalize(im, nomr1, norm2)

        im_f = gaussian(im_n, sigma=sigma)


        ori_main, max_evec, min_evec, max_eval, min_eval = analyze_area(im_f, mask)



        res.append((ori_main,im_name))
        if sigma_test:
            oris = []
            sigmas = [0.2,0.3,0.5,0.8,1,1.5,2,3,4,5,6,8,10,15]
            for sig in tqdm(sigmas):
                im_fl = gaussian(im_n, sigma=sig)
                ori, max_evec, min_evec, max_eval, min_eval = analyze_area(im_fl, mask)
                oris.append(ori)
            fig = plt.figure()
            plt.plot(sigmas, oris)
            fig.savefig(os.path.join(out_folder, im_name+"_sigma_test.svg"))

        ori_list, angs = analyze_area_full_orientation(im_f, mask, points=100, length=np.pi * 2)

        # ot_xx, ot_yx, ot_yy = get_structure_tensor_roi(im_f, mask=mask)
        # max_evec, min_evec, max_eval, min_eval = get_principal_vectors(ot_xx, ot_yx, ot_yy)
        # ori = (max_eval - min_eval) / (max_eval + min_eval)

        oris.append(ori_main)
        if plot:
            plot1(im, im_f, sigma, out_folder, [max_evec, min_evec, max_eval, min_eval, ori_main],mask, name=im_name+"_.svg")
            #ori_list, angs = analyze_area_full_orientation(im_f, mask, points=100, length=np.pi * 2)
            full_angle_plot(ori_list, angs, out_folder, name=im_name+"_orient_dist.svg")

    with open(os.path.join(out_folder,"res.txt"),"w") as f:
        for r in res:
            f.write(str(np.round(r[0], 3)) + "\t" + r[1] + "\n")



