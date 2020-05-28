'''
trying to apply structure tensor locally --> coud be used to "filter out all non coherent== non oriented structures"

'''

from fo_testing3 import *



if __name__ == "__main__":

    folder = "/home/user/Desktop/ingo_fiber_orientations/"



    db = clickpoints.DataFile(os.path.join(folder, "db.cdb"))

    out_folder = "/home/user/Desktop/ingo_fiber_orientations/sigma_test2"
    createFolder(out_folder)
    i = db.getImages()[5]
    im = i.data
    mask = db.getMask(image=i).data
    mask = binary_fill_holes(mask)
    '''
    for j, i in enumerate(db.getImages()):
        im = i.data
        plt.figure()
        plt.imshow(im)
    '''
    im = plt.imread("/home/user/Desktop/fibre_orientation_test/pure_grid-0.png")


    plt.figure()
    for sigma2 in tqdm([1,2,3,4,6,8,12,14,16,20]):


        sigma1 = 1
        # image preprocessing
        # sigma2 = 20  # wnidow size of finding local coherency
        im_f = gaussian(im, sigma=sigma1)
        circle(r=50, c=im.shape[1] - 50, radius=sigma2, shape=im.shape)
        circ = np.zeros(im.shape) + np.nan
        circ[circle(r=50, c=im.shape[1] - 50, radius=sigma2, shape=im.shape)] = 1

        ori, max_evec, min_evec, max_eval, min_eval = analyze_local(im_f, sigma=sigma2,size=50, filter_type="uniform")
        ori, max_evec, min_evec, max_eval, min_eval = analyze_local(im_f, sigma=15, size=50, filter_type="gaussian")
        ori, max_evec, min_evec, max_eval, min_eval = analyze_local_2_filters(im_f, sigma2, sigma2=0, window_size=40, filter_type="uniform")

        #min_evec


        if plot:
            plt.figure()
            plt.imshow(im)
            plt.gca().imshow(circ, cmap="spring", vmin=0, vmax=1)

            plt.figure()
            plt.imshow(im_f)
            plt.gca().imshow(circ, cmap="spring", vmin=0, vmax=1)

            plt.figure()
            plt.imshow(ori)
            plt.colorbar()
            f = np.nanpercentile(ori, 75)
            print("cohernecy threshold=", f)
            fig, ax = show_quiver(min_evec[:, :, 0] * ori, min_evec[:, :, 1] * ori, filter=[f, 12],
                                  scale_factor=0.1,
                                  width=0.003, cbar_str="coherency", cmap="viridis")

            ax.imshow(circ, cmap="spring", vmin=0, vmax=1)


    ori_list, angs = analyze_area_full_orientation(im_f, mask, points=100, length=np.pi * 2)
    # ot_xx, ot_yx, ot_yy = get_structure_tensor_roi(im_f, mask=mask)
    # max_evec, min_evec, max_eval, min_eval = get_principal_vectors(ot_xx, ot_yx, ot_yy)
    # ori = (max_eval - min_eval) / (max_eval + min_eval)
    #oris.append(ori)

    plot1(im, im_f, sigma1, out_folder, [max_evec, min_evec, max_eval, min_eval])
    ori_list, angs = analyze_area_full_orientation(im_f, mask, points=100, length=np.pi * 2)
    full_angle_plot(ori_list, angs, out_folder, sigma=str(sigma1))

