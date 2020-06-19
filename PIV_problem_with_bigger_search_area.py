
import matplotlib.pyplot as plt
from openpiv.pyprocess import *


def correlate_windows(window_a, window_b, corr_method='fft', nfftx=None, nffty=None):

    if corr_method == 'fft':
        window_b = np.conj(window_b[::-1, ::-1])
        if nfftx is None:
            nfftx = nextpower2(window_b.shape[0] + window_a.shape[0])
        if nffty is None:
            nffty = nextpower2(window_b.shape[1] + window_a.shape[1])

        f2a = rfft2(normalize_intensity(window_a), s=(nfftx, nffty))
        f2b = rfft2(normalize_intensity(window_b), s=(nfftx, nffty))
        corr = irfft2(f2a * f2b).real
        ###### full correlation matrix has window_a.shape + window_b.shape - 1 dimensions ####
        ### simple example: lets correlate 1d arrays: [1,2,3] (shape=3) with [3,4] (shape=2)
        ### there are 4 conolution steps: [01230]    [01230]    [01230]    [01230]
        #                                    *          *          *          *
        #                                 [34000]    [03400]    [00340]   [00034]

        corr = corr[:window_a.shape[0] + window_b.shape[0] - 1,
               :window_b.shape[1] + window_a.shape[1] - 1]
        return corr


    elif corr_method == 'direct':
        return convolve2d(normalize_intensity(window_a),
                          normalize_intensity(window_b[::-1, ::-1]), 'full')
    else:
        raise ValueError('method is not implemented')



def check_search_area_size(search_area_size, window_size, warn=True):
    # TODO: discuss if this is correct
    # displacement between window and search area can only be uniquely defined if size of the correlation matrix - window_size is
    # divisibale by 2. This function warns, and corrects search_area_size by adding +1  if this is not the case.
    #
    corr_size = search_area_size + window_size - 1  # size of the correlation field
    if (corr_size - window_size) % 2 != 0:
            search_area_size += 1
            if warn:
                print(
                    "search_area_size at position does not match windowsize; increased search_area_size by 1")

    return search_area_size

def extended_search_area_piv_corrected( frame_a, frame_b,
            window_size,
            overlap=0,
            dt=1.0,
            search_area_size=None,
            corr_method='fft',
            subpixel_method='gaussian',
            sig2noise_method=None,
            width=2,
            nfftx=None, nffty=None):




        if search_area_size == 0:
            search_area_size = window_size
        search_area_size = check_search_area_size(search_area_size, window_size, warn=True)
        if overlap >= window_size:
            raise ValueError('Overlap has to be smaller than the window_size')

        if search_area_size < window_size:
            raise ValueError('Search size cannot be smaller than the window_size')

        if (window_size > frame_a.shape[0]) or (window_size > frame_a.shape[1]):
            raise ValueError('window size cannot be larger than the image')

        # get field shape
        n_rows, n_cols = get_field_shape((frame_a.shape[0], frame_a.shape[1]),
                                         window_size, overlap)

        sig2noise = np.zeros((n_rows, n_cols))

        u, v = np.zeros((n_rows, n_cols)), np.zeros((n_rows, n_cols))

        # if we want sig2noise information, allocate memory
        if sig2noise_method is not None:
            sig2noise = np.zeros((n_rows, n_cols))

        # loop over the interrogation windows
        # i, j are the row, column indices of the center of each interrogation
        # window
        for k in range(n_rows):
            # range(range(search_area_size/2, frame_a.shape[0] - search_area_size/2, window_size - overlap ):
            for m in range(n_cols):
                # range(search_area_size/2, frame_a.shape[1] - search_area_size/2 , window_size - overlap ):

                # Select first the largest window, work like usual from the top left corner
                # the left edge goes as:
                # e.g. 0, (search_area_size - overlap), 2*(search_area_size - overlap),....

                il = k * (search_area_size - overlap)
                ir = il + search_area_size

                # same for top-bottom
                jt = m * (search_area_size - overlap)
                jb = jt + search_area_size

                # pick up the window in the second image
                window_b = frame_b[il:ir, jt:jb]

                # now shift the left corner of the smaller window inside the larger one
                il += (search_area_size - window_size) // 2
                # and it's right side is just a window_size apart
                ir = il + window_size
                # same same
                jt += (search_area_size - window_size) // 2
                jb = jt + window_size

                window_a = frame_a[il:ir, jt:jb]

                print(window_b.shape, window_a.shape)
                '''
                # fig,axs = plt.subplots(1,2);
                plt.figure()
                window_b_show = window_b.copy()
                window_b_show[window_b_show == 0] = np.nan

                window_a_show = np.zeros(window_b_show.shape)
                window_a_show[
                int((window_b.shape[0] - window_a.shape[0]) / 2):-int((window_b.shape[0] - window_a.shape[0]) / 2),
                int((window_b.shape[1] - window_a.shape[1]) / 2):-int(
                    (window_b.shape[1] - window_a.shape[1]) / 2)] = window_a
                window_a_show[window_a_show == 0] = np.nan

                plt.imshow(window_a_show, alpha=0.5, cmap="rainbow")
                plt.imshow(window_b_show, alpha=0.5, cmap="viridis")
                '''
                if np.any(window_a):
                    corr = correlate_windows(window_a, window_b,
                                             corr_method=corr_method,
                                             nfftx=nfftx, nffty=nffty)
                    #                 plt.figure()
                    #                 plt.contourf(corr)
                    #                 plt.show()
                    # get subpixel approximation for peak position row and column index
                    row, col = find_subpixel_peak_position(corr, subpixel_method=subpixel_method)



                    ############# hopefully correct identification of displacements ############
                    # identifying the center of the correlation matrix
                    corr_center = (np.array(corr.shape)-1) / 2
                    # calculating the displacement form the center to the maximum of the correlation matrix
                    row = corr_center[0] - row
                    col = corr_center[1] - col
                    #plt.text(1, 1, "row=" + str(np.round(row, 2)) + "col=" + str(np.round(col)))
                    # get displacements, apply coordinate system definition
                    u[k, m], v[k, m] = -col, row

                    # get signal to noise ratio
                    if sig2noise_method is not None:
                        sig2noise[k, m] = sig2noise_ratio(
                            corr, sig2noise_method=sig2noise_method, width=width)

        # return output depending if user wanted sig2noise information
        if sig2noise_method is not None:
            return u / dt, v / dt, sig2noise
        else:
            return u / dt, v / dt


# genrating a bar shifting it by 1 pixel
size = (100, 100)
shape1 = np.zeros(size)
shape2 = np.zeros(size)
shape1[50, 50:58] = 1
shape2[50, 50:59] = 1

window_size = 5
overlap = 4
##
plt.close("all")
## standard extende_search_area function with window_size == search_area
search_area = 5
u, v, sig2noise = extended_search_area_piv(shape1, shape2, window_size=window_size, overlap=overlap, search_area_size=search_area, subpixel_method='gaussian',
                              sig2noise_method='peak2peak', corr_method="fft",
                              width=2)
## standard extende_search_area function with window_size > search_area
search_area = 7
u1, v1, sig2noise1 = extended_search_area_piv(shape1, shape2, window_size=window_size, overlap=overlap, search_area_size=search_area, subpixel_method='gaussian',
                              sig2noise_method='peak2peak', corr_method="fft",
                              width=2)


##corrected extende_search_area function with window_size > search_area
window_size = 7
search_area = 7
u2, v2, sig2noise2 = extended_search_area_piv(shape1, shape2, window_size=window_size, overlap=overlap, search_area_size=search_area, subpixel_method='gaussian',
                              sig2noise_method='peak2peak', corr_method="fft",
                              width=2)





plt.figure()
plt.imshow(shape1)
plt.figure()
plt.imshow(shape2)

plt.figure()
plt.quiver(-u, v)
plt.imshow(np.sqrt(u**2+v**2))
plt.figure()
plt.quiver(-u1, v1)
plt.imshow(np.sqrt(u1**2+v1**2))
plt.figure()
plt.quiver(-u2, v2)
plt.imshow(np.sqrt(u2**2+v2**2))

############################# there is something "wrong" with how search area is selected ##################################