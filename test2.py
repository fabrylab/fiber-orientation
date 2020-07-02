from openpiv import tools, process, scaling, pyprocess, validation, filters
from openpiv.process import *
import numpy as np
import pylab
# %matplotlib inline

# %%
frame_a  = tools.imread( '../openpiv-python/openpiv/examples/test1/exp1_001_a.bmp' )
frame_b  = tools.imread( '../openpiv-python/openpiv/examples/test1/exp1_001_b.bmp' )
#pylab.imshow(np.c_[frame_a,np.ones((frame_a.shape[0],20)),frame_b],cmap=pylab.cm.gray)

# %%
# %%time


def ep_local(frame_a,frame_b,window_size,overlap = 0, dt = 1.0, search_area_size = 0, subpixel_method = 'gaussian', sig2noise_method = None, width = 2, nfftx = 0, nffty = 0):
    if search_area_size == 0:
        search_area_size = window_size

    if overlap >= window_size:
        raise ValueError('Overlap has to be smaller than the window_size')

    if search_area_size < window_size:
        raise ValueError('Search size cannot be smaller than the window_size')

    if (window_size > frame_a.shape[0]) or (window_size > frame_a.shape[1]):
        raise ValueError('window size cannot be larger than the image')

    # get field shape
    n_rows, n_cols = get_field_shape((frame_a.shape[0], frame_a.shape[1]), window_size, overlap)
    # define arrays
    window_a = np.zeros([window_size, window_size], dtype=DTYPEi)
    search_area = np.zeros([search_area_size, search_area_size], dtype=DTYPEi)
    corr = np.zeros([search_area_size, search_area_size], dtype=DTYPEf)
    u = np.zeros([n_rows, n_cols], dtype=DTYPEf)
    v = np.zeros([n_rows, n_cols], dtype=DTYPEf)
    sig2noise = np.zeros([n_rows, n_cols], dtype=DTYPEf)

    # loop over the interrogation windows
    # i, j are the row, column indices of the top left corner
    I = 0
    for i in range(0, frame_a.shape[0] - window_size + 1, window_size - overlap):
        J = 0
        for j in range(0, frame_a.shape[1] - window_size + 1, window_size - overlap):

            # get interrogation window matrix from frame a
            for k in range(window_size):
                for l in range(window_size):
                    window_a[k, l] = frame_a[i + k, j + l]

            # get search area using frame b
            for k in range(search_area_size):
                for l in range(search_area_size):

                    # fill with zeros if we are out of the borders
                    if i + window_size / 2 - search_area_size // 2 + k < 0 or \
                            i + window_size // 2 - search_area_size // 2 + k >= frame_b.shape[0]:
                        search_area[k, l] = 0
                    elif j + window_size // 2 - search_area_size // 2 + l < 0 or \
                            j + window_size // 2 - search_area_size // 2 + l >= frame_b.shape[1]:
                        search_area[k, l] = 0
                    else:
                        search_area[k, l] = frame_b[i + window_size // 2 - search_area_size // 2 + k,
                                                    j + window_size // 2 - search_area_size // 2 + l]

            # compute correlation map
            if any(window_a.flatten()):
                corr = correlate_windows(search_area, window_a, nfftx=nfftx, nffty=nffty)
                c = CorrelationFunction(corr)

                # find subpixel approximation of the peak center
                i_peak, j_peak = c.subpixel_peak_position(subpixel_method)

                # velocities
                v[I, J] = -((i_peak - corr.shape[0] / 2) - (search_area_size - window_size) / 2) / dt
                u[I, J] = ((j_peak - corr.shape[0] / 2) - (search_area_size - window_size) / 2) / dt

                # compute signal to noise ratio
                if sig2noise_method:
                    sig2noise[I, J] = c.sig2noise_ratio(sig2noise_method, width)
            else:
                v[I, J] = 0.0
                u[I, J] = 0.0
                # compute signal to noise ratio
                if sig2noise_method:
                    sig2noise[I, J] = np.inf

            # go to next vector
            J = J + 1

        # go to next vector
        I = I + 1

    if sig2noise_method:
        return u, v, sig2noise
    else:
        return u, v




u, v, sig2noise = ep_local( frame_a.astype(np.int32), frame_b.astype(np.int32),
                                                   window_size=23, overlap=12, dt=0.02,
                                                   search_area_size=64,
                                                   sig2noise_method='peak2peak' )
x, y = process.get_coordinates( image_size=frame_a.shape, window_size=23, overlap=12 )
u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 2.5 )
u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
tools.save(x, y, u, v, mask, 'exp1_001_extended.txt' )

