"""This module contains a pure python implementation of the basic
cross-correlation algorithm for PIV image processing."""

__licence_ = """
Copyright (C) 2011  www.openpiv.net

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy.lib.stride_tricks
import numpy as np
from numpy.fft import rfftn, irfftn
from numpy import ma
from scipy.signal import convolve2d
from scipy.signal import convolve2d
from numpy import log
from scipy.signal import correlate



def check_search_area_size(search_area_size, window_size, warn=True):
    # TODO: discuss if this is correct
    # displacement between window and searcharea can only be uniquely defined if window_size + searcharea_size is an even number
    # this warns, and corrects search_area_size if this is not the case
    corr_size = np.array(search_area_size) + np.array(window_size) - 1  # size of the correlation field
    search_area_size_corrected = []
    for i, c in enumerate(corr_size):
        if c % 2 == 0:  # if this is the case the blocks of window and searcharea have no uniequely defined alignement
            search_area_size_corrected.append(search_area_size[i] + 1)
            if warn:
                print(
                    "search_area_size at position %s does not match windowsize; increased search_area_size by 1" % str(
                        i))
        else:
            search_area_size_corrected.append(search_area_size[i])

    return search_area_size_corrected

def get_coordinates(image_size, window_size, overlap):
    """Compute the x, y coordinates of the centers of the interrogation windows.

    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.

    window_size: int
        the size of the interrogation windows.

    overlap: int
        the number of pixel by which two adjacent interrogation
        windows overlap.


    Returns
    -------
    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.

    y : 2d np.ndarray
        a two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.

    """

    # get shape of the resulting flow field
    field_shape = get_field_shape(image_size, window_size, overlap)

    # compute grid coordinates of the interrogation window centers
    # compute grid coordinates of the interrogation window centers
    z = np.arange(field_shape[2]) * (window_size[2] - overlap[2]) + window_size[2] / 2.0
    x = np.arange(field_shape[1]) * (window_size[1] - overlap[1]) + window_size[1] / 2.0
    y = np.arange(field_shape[0]) * (window_size[0] - overlap[0]) + window_size[0] / 2.0

    return np.meshgrid(x, y, z)


def get_field_shape(image_size, window_size, overlap):
    """Compute the shape of the resulting flow field.

    Given the image size, the interrogation window size and
    the overlap size, it is possible to calculate the number
    of rows and columns of the resulting flow field.

    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.

    window_size: int
        the size of the interrogation window.

    overlap: int
        the number of pixel by which two adjacent interrogation
        windows overlap.


    Returns
    -------
    field_shape : two elements tuple
        the shape of the resulting flow field
    """

    return (np.array(image_size)-np.array(window_size)) // (np.array(window_size) - np.array(overlap)) + 1



def moving_window_array(array, window_size, overlap):
    """
    This is a nice numpy trick. The concept of numpy strides should be
    clear to understand this code.

    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in which
    each slice, (along the first axis) is an interrogation window.

    """
    sz = array.itemsize
    shape = array.shape
    array = np.ascontiguousarray(array)

    strides = (sz * shape[1] * (window_size - overlap),
               sz * (window_size - overlap), sz * shape[1], sz)
    shape = (int((shape[0] - window_size) / (window_size - overlap)) + 1, int(
        (shape[1] - window_size) / (window_size - overlap)) + 1, window_size, window_size)

    return numpy.lib.stride_tricks.as_strided(array, strides=strides, shape=shape).reshape(-1, window_size, window_size)


def find_first_peak(corr):
    """
    Find row and column indices of the first correlation peak.

    Parameters
    ----------
    corr : np.ndarray
        the correlation map

    Returns
    -------

    """

    return np.unravel_index(np.argmax(corr), corr.shape), corr.max()


def find_second_peak(corr, i=None, j=None, z=None, width=2):
    """
    Find the value of the second largest peak.

    The second largest peak is the height of the peak in
    the region outside a 3x3 submatrix around the first
    correlation peak.

    Parameters
    ----------
    corr: np.ndarray
          the correlation map.

    i,j : ints
          row and column location of the first peak.

    width : int
        the half size of the region around the first correlation
        peak to ignore for finding the second peak.

    Returns
    -------
    i : int
        the row index of the second correlation peak.

    j : int
        the column index of the second correlation peak.

    z : int
        the 3rd index of the second correlation peak.


    corr_max2 : int
        the value of the second correlation peak.

    """

    if i is None or j is None or z is None:
        (i, j, z), tmp = find_first_peak(corr)    # TODO: why tmp?

    # create a masked view of the corr
    tmp = corr.view(ma.MaskedArray)

    # set width x width square submatrix around the first correlation peak as masked.
    # Before check if we are not too close to the boundaries, otherwise we
    # have negative indices
    iini = max(0, i - width)
    ifin = min(i + width + 1, corr.shape[0])
    jini = max(0, j - width)
    jfin = min(j + width + 1, corr.shape[1])
    zini = max(0, z - width)
    zfin = min( z + width +1, corr.shape[1])

    tmp[iini:ifin, jini:jfin, zini:zfin] = ma.masked
    (i, j, z), corr_max2 = find_first_peak(tmp)

    return (i, j, z), corr_max2


def find_subpixel_peak_position(corr, subpixel_method='gaussian'):
    """
    Find subpixel approximation of the correlation peak.

    This function returns a subpixels approximation of the correlation
    peak by using one of the several methods available. If requested,
    the function also returns the signal to noise ratio level evaluated
    from the correlation map.

    Parameters
    ----------
    corr : np.ndarray
        the correlation map.

    subpixel_method : string
         one of the following methods to estimate subpixel location of the peak:
         'centroid' [replaces default if correlation map is negative],
         'gaussian' [default if correlation map is positive],
         'parabolic'.

    Returns
    -------
    subp_peak_position : two elements tuple
        the fractional row and column indices for the sub-pixel
        approximation of the correlation peak.
    """

    # initialization
    # default_peak_position = (np.floor(corr.shape[0] / 2.), np.floor(corr.shape[1] / 2.))
    default_peak_position = (0, 0, 0)

    # the peak locations
    (peak1_i, peak1_j, peak1_z), dummy = find_first_peak(corr)

    try:
        # the peak and its neighbours: left, right, down, up
        c = corr[peak1_i, peak1_j, peak1_z]
        cl = corr[peak1_i - 1, peak1_j, peak1_z]
        cr = corr[peak1_i + 1, peak1_j, peak1_z]
        cd = corr[peak1_i, peak1_j - 1, peak1_z]
        cu = corr[peak1_i, peak1_j + 1, peak1_z]
        cf = corr[peak1_i, peak1_j, peak1_z - 1]
        cb = corr[peak1_i, peak1_j, peak1_z + 1]


        # gaussian fit
        if np.any(np.array([c, cl, cr, cd, cu, cf, cb]) < 0) and subpixel_method == 'gaussian':
            subpixel_method = 'centroid'

        try:
            if subpixel_method == 'centroid':
                subp_peak_position = (
                    ((peak1_i - 1) * cl + peak1_i * c + (peak1_i + 1) * cr) / (cl + c + cr),
                    ((peak1_j - 1) * cd + peak1_j * c + (peak1_j + 1) * cu) / (cd + c + cu),
                    ((peak1_z - 1) * cf + peak1_z * c + (peak1_z + 1) * cb) / (cf + c + cb))

            elif subpixel_method == 'gaussian':
                subp_peak_position = (
                    peak1_i + ((np.log(cl) - np.log(cr)) / (2 * np.log(cl) - 4 * np.log(c) + 2 * np.log(cr))),
                    peak1_j + ((np.log(cd) - np.log(cu)) / (2 * np.log(cd) - 4 * np.log(c) + 2 * np.log(cu))),
                    peak1_z + ((np.log(cf) - np.log(cb)) / (2 * np.log(cf) - 4 * np.log(c) + 2 * np.log(cb)))
                )

            elif subpixel_method == 'parabolic':
                subp_peak_position = (peak1_i + (cl - cr) / (2 * cl - 4 * c + 2 * cr),
                                      peak1_j + (cd - cu) / (2 * cd - 4 * c + 2 * cu),
                                      peak1_z + (cf - cb) / (2 * cf - 4 * c + 2 * cb))
        except:
            subp_peak_position = default_peak_position

    except IndexError:
        subp_peak_position = default_peak_position # TODO: is this a good idea??

    return np.array(subp_peak_position) - np.array(default_peak_position)


def sig2noise_ratio(corr, sig2noise_method='peak2peak', width=2):
    """
    Computes the signal to noise ratio from the correlation map.

    The signal to noise ratio is computed from the correlation map with
    one of two available method. It is a measure of the quality of the
    matching between to interogation windows.

    Parameters
    ----------
    corr : 2d np.ndarray
        the correlation map.

    sig2noise_method: string
        the method for evaluating the signal to noise ratio value from
        the correlation map. Can be `peak2peak`, `peak2mean` or None
        if no evaluation should be made.

    width : int, optional
        the half size of the region around the first
        correlation peak to ignore for finding the second
        peak. [default: 2]. Only used if ``sig2noise_method==peak2peak``.

    Returns
    -------
    sig2noise : float
        the signal to noise ratio from the correlation map.

    """

    # compute first peak position
    (peak1_i, peak1_j, peak1_z), corr_max1 = find_first_peak(corr)

    # now compute signal to noise ratio
    if sig2noise_method == 'peak2peak':
        # find second peak height
        (peak1_i, peak1_j, peak1_z), corr_max2 = find_second_peak(
            corr, peak1_i, peak1_j, peak1_z, width=width)

        # if it's an empty interrogation window
        # if the image is lacking particles, totally black it will correlate to very low value, but not zero
        # if the first peak is on the borders, the correlation map is also wrong
        if corr_max1 < 1e-3 or any([x==0 or x==corr.shape[i] for i,x in enumerate([peak1_i,peak1_j, peak1_z])]):
            return 0.0

    elif sig2noise_method == 'peak2mean':
        # find mean of the correlation map
        corr_max2 = corr.mean()

    else:
        raise ValueError('wrong sig2noise_method')

    # avoid dividing by zero
    try:
        sig2noise = corr_max1 / corr_max2
    except ValueError:
        sig2noise = np.inf

    return sig2noise


def correlate_windows(window_a, window_b, corr_method='fft', nfftx=None, nffty=None, nfftz=None):
    """Compute correlation function between two interrogation windows.

    The correlation function can be computed by using the correlation
    theorem to speed up the computation.

    Parameters
    ----------
    window_a : 2d np.ndarray
        a two dimensions array for the first interrogation window,

    window_b : 2d np.ndarray
        a two dimensions array for the second interrogation window.

    corr_method   : string
        one of the two methods currently implemented: 'fft' or 'direct'.
        Default is 'fft', which is much faster.

    nfftx   : int
        the size of the 2D FFT in x-direction,
        [default: 2 x windows_a.shape[0] is recommended].

    nffty   : int
        the size of the 2D FFT in y-direction,
        [default: 2 x windows_a.shape[1] is recommended].


    Returns
    -------
    corr : 2d np.ndarray
        a two dimensions array for the correlation function.

    Note that due to the wish to use 2^N windows for faster FFT
    we use a slightly different convention for the size of the
    correlation map. The theory says it is M+N-1, and the
    'direct' method gets this size out
    the FFT-based method returns M+N size out, where M is the window_size
    and N is the search_area_size
    It leads to inconsistency of the output
    """

    if corr_method == 'fft':
        window_b = np.conj(window_b[::-1, ::-1, ::-1])
        if nfftx is None:
            nfftx = nextpower2(window_b.shape[0] + window_a.shape[0])
        if nffty is None:
            nffty = nextpower2(window_b.shape[1] + window_a.shape[1])
        if nfftz is None:
            nfftz = nextpower2(window_b.shape[2] + window_a.shape[2])

        f2a = rfftn(normalize_intensity(window_a), s=(nfftx, nffty,nfftz))
        f2b = rfftn(normalize_intensity(window_b), s=(nfftx, nffty,nfftz))
        corr = irfftn(f2a * f2b).real
        corr = corr[:window_a.shape[0] + window_b.shape[0] ,
               :window_b.shape[1] + window_a.shape[1] ,:window_b.shape[2] + window_a.shape[2]]
        return corr
    elif corr_method == 'direct':
        return convolve2d(normalize_intensity(window_a),
                          normalize_intensity(window_b[::-1, ::-1, ::-1]), 'full')
    else:
        raise ValueError('method is not implemented')


def normalize_intensity(window):
    """Normalize interrogation window by removing the mean value.

    Parameters
    ----------
    window :  2d np.ndarray
        the interrogation window array

    Returns
    -------
    window :  2d np.ndarray
        the interrogation window array, with mean value equal to zero.

    """
    return window - window.mean()


def extended_search_area_piv3D(
        frame_a, frame_b,
        window_size,
        overlap=0,
        dt=(1.0,1.0,1.0),
        search_area_size=None,
        corr_method='fft',
        subpixel_method='gaussian',
        sig2noise_method=None,
        width=2,
        nfftx=None, nffty=None,nfftz=None):
    """Standard PIV cross-correlation algorithm, with an option for
    extended area search that increased dynamic range. The search region
    in the second frame is larger than the interrogation window size in the
    first frame. For Cython implementation see
    openpiv.process.extended_search_area_piv

    This is a pure python implementation of the standard PIV cross-correlation
    algorithm. It is a zero order displacement predictor, and no iterative process
    is performed.

    Parameters
    ----------
    frame_a : 2d np.ndarray
        an two dimensions array of integers containing grey levels of
        the first frame.

    frame_b : 2d np.ndarray
        an two dimensions array of integers containing grey levels of
        the second frame.

    window_size : int
        the size of the (square) interrogation window, [default: 32 pix].

    overlap : int
        the number of pixels by which two adjacent windows overlap
        [default: 16 pix].

    dt : float
        the time delay separating the two frames [default: 1.0].

    corr_method : string
        one of the two methods implemented: 'fft' or 'direct',
        [default: 'fft'].

    subpixel_method : string
         one of the following methods to estimate subpixel location of the peak:
         'centroid' [replaces default if correlation map is negative],
         'gaussian' [default if correlation map is positive],
         'parabolic'.

    sig2noise_method : string
        defines the method of signal-to-noise-ratio measure,
        ('peak2peak' or 'peak2mean'. If None, no measure is performed.)

    nfftx   : int
        the size of the 2D FFT in x-direction,
        [default: 2 x windows_a.shape[0] is recommended]

    nffty   : int
        the size of the 2D FFT in y-direction,
        [default: 2 x windows_a.shape[1] is recommended]

    width : int
        the half size of the region around the first
        correlation peak to ignore for finding the second
        peak. [default: 2]. Only used if ``sig2noise_method==peak2peak``.

    search_area_size : int
       the size of the interrogation window in the second frame,
       default is the same interrogation window size and it is a
       fallback to the simplest FFT based PIV


    Returns
    -------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        in pixels/seconds.

    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        in pixels/seconds.

    sig2noise : 2d np.ndarray, ( optional: only if sig2noise_method is not None )
        a two dimensional array the signal to noise ratio for each
        window pair.

    """



    # checking if the input is correct
    window_size, overlap, search_area_size = check_input(window_size, overlap, search_area_size, frame_a, frame_b)
    print(window_size, overlap, search_area_size)
    # get field shape
    field_shape = get_field_shape((frame_a.shape),window_size, overlap)

    u = np.zeros(field_shape)
    v = np.zeros(field_shape)
    w = np.zeros(field_shape)

    # if we want sig2noise information, allocate memory
    if sig2noise_method is not None:
        sig2noise = np.zeros(field_shape)

    # loop over the interrogation windows
    # i, j are the row, column indices of the center of each interrogation
    # window
    for k in range(field_shape[0]):
        # range(range(search_area_size/2, frame_a.shape[0] - search_area_size/2, window_size - overlap ):
        for m in range(field_shape[1]):
            # range(search_area_size/2, frame_a.shape[1] - search_area_size/2 , window_size - overlap ):
            for l in range(field_shape[2]):
                # Select first the largest window, work like usual from the top left corner
                # the left edge goes as:
                # e.g. 0, (search_area_size - overlap), 2*(search_area_size - overlap),....

                il = k * (search_area_size[0] - overlap[0])
                ir = il + search_area_size[0]

                # same for top-bottom
                jt = m * (search_area_size[1] - overlap[1])
                jb = jt + search_area_size[1]

                # same for front-back
                jf = l * (search_area_size[2] - overlap[2])
                jba = jf + search_area_size[2]

                # pick up the window in the second image
                window_b = frame_b[il:ir, jt:jb, jf:jba]

                # now shift the left corner of the smaller window inside the larger one
                il += (search_area_size[0] - window_size[0]) // 2
                # and it's right side is just a window_size apart
                ir = il + window_size[0]
                # same same
                jt += (search_area_size[1] - window_size[1]) // 2
                jb = jt + window_size[1]
                jf += (search_area_size[2] - window_size[2]) // 2
                jba = jf + window_size[2]

                window_a = frame_a[il:ir, jt:jb, jf:jba]

                if np.any(window_a) and np.any(window_b):
                    corr = correlate_windows(window_a, window_b,
                                             corr_method=corr_method,
                                             nfftx=nfftx, nffty=nffty, nfftz=nfftz)

                    # get subpixel approximation for peak position row and column index
                    row, col, z = find_subpixel_peak_position(corr, subpixel_method=subpixel_method)

                    row -= (search_area_size[0] + window_size[0] - 1) // 2
                    col -= (search_area_size[1] + window_size[1] - 1) // 2
                    z -= (search_area_size[2] + window_size[2] - 1) // 2

                    # get displacements, apply coordinate system definition
                    u[k, m, l], v[k, m, l], w[k, m, l] = -col, row, z
                    ## TODO: why we return -col ?

                    # get signal to noise ratio
                    if sig2noise_method is not None:
                        sig2noise[k, m, l] = sig2noise_ratio(
                            corr, sig2noise_method=sig2noise_method, width=width)



    # return output if user wanted sig2noise information
    if sig2noise_method is not None:
        return u / dt[0], v / dt[1], w/dt[2], sig2noise
    else:
        return u / dt[0], v / dt[1], w/dt[2]


def nextpower2(i):
    """ Find 2^n that is equal to or greater than. """
    n = 1
    while n < i: n *= 2
    return n

def check_input(window_size, overlap, search_area_size, frame_a ,frame_b):
    # check the inputs for validity
    search_area_size = [ws if x == 0 or x is None else x for x, ws in zip(search_area_size, window_size)]
    # search_area_size + window_size should be even number, else deformation cannot be defined uniquely//
    # correcting search_area_size by +1 if necessary
    search_area_size = check_search_area_size(search_area_size, window_size, warn=True)

    if any((np.array(window_size) - np.array(overlap)) < 0):
        raise ValueError('Overlap has to be smaller than the window_size')

    if any((np.array(search_area_size) - np.array(window_size)) < 0):
        raise ValueError('Search size cannot be smaller than the window_size')

    if any([ws > ims for ws, ims in zip(window_size, frame_a.shape)]):
        raise ValueError('window size cannot be larger than the image')

    if any([ims_a != ims_b for ims_a, ims_b in zip(frame_a.shape, frame_b.shape)]):
        raise ValueError('frame a and frame b have different sizes.')

    return window_size, overlap, search_area_size



### TODO: implement dis
def check_search_area_size(search_area_size, window_size, warn=True):
    # dsiplacement between window and searcharea can only be uniquely defined if window_size + searcharea_size is an even number
    # this warns, and corrects search_area_size if this is not the case
    corr_size = np.array(search_area_size) + np.array(window_size) - 1  # size of the correlation field
    search_area_size_corrected = []
    for i, c in enumerate(corr_size):
        if c % 2 == 0:  # if this is the case the blocks of window and searcharea have no uniequely defined alignement
            search_area_size_corrected.append(search_area_size[i] + 1)
            if warn:
                print(
                    "search_area_size at position %s does not match windowsize; increased search_area_size by 1" % str(
                        i))
        else:
            search_area_size_corrected.append(search_area_size[i])

    return search_area_size_corrected






def replace_nans(a, max_iter, tol, kernel_size = 2, method = 'disk'):


    DTYPEf = np.float
    DTYPEi = np.int

    array = a.copy()
    n_dim = len(array.shape)

    # generating the kernel
    kernel = np.zeros([2 * kernel_size + 1] * len(array.shape), dtype=int)
    if method == 'localmean':
        kernel += 1
    elif method == 'disk':
        dist, dist_inv = get_dist(kernel, kernel_size)
        kernel[dist <= kernel_size] = 1
    elif method == 'distance':
        dist, dist_inv = get_dist(kernel, kernel_size)
        kernel[dist <= kernel_size] = dist_inv[dist <= kernel_size]
    else:
        raise ValueError('method not valid. Should be one of `localmean`, `disk` or `distance`.')

    # list of kernel array indices
    kernel_indices = np.indices(kernel.shape)
    kernel_indices = np.reshape(kernel_indices, (n_dim, (2 * kernel_size + 1) ** n_dim), order="C").T

    # indices where array is NaN
    nan_indices = np.array(np.nonzero(np.isnan(array))).T.astype(DTYPEi)

    # number of NaN elements
    n_nans = len(nan_indices)

    # arrays which contain replaced values to check for convergence
    replaced_new = np.zeros(n_nans, dtype=DTYPEf)
    replaced_old = np.zeros(n_nans, dtype=DTYPEf)

    # make several passes
    # until we reach convergence
    for it in range(max_iter):
        # note: identifying new nan indices and looping other the new indices would give slightly different result

        # for each NaN element
        for k in range(n_nans):
            ind = nan_indices[k] #2 or 3 indices indicating the position of a nan element
            # init to 0.0
            replaced_new[k] = 0.0
            n = 0.0

            # generating a list of indices of the convolution window in the array
            slice_indices = np.array(np.meshgrid(*[range(i-kernel_size,i+kernel_size+1) for i in ind]))
            slice_indices = np.reshape(slice_indices,( n_dim, (2 * kernel_size + 1) ** n_dim), order="C").T

            # loop over the kernel
            for s_index, k_index in zip(slice_indices, kernel_indices):
                s_index = tuple(s_index) # this is necessary for numpy array indexing
                k_index = tuple(k_index)

                # skip if we are outside of array boundaries, if the array element is nan or if the kernel element is zero
                if all([s >= 0 and s < bound for s, bound  in zip(s_index, array.shape)]):
                    if not np.isnan(array[s_index]) and kernel[k_index] != 0:
                    # convolve kernel with original array
                        replaced_new[k] = replaced_new[k] + array[s_index] * kernel[k_index]
                        n = n + kernel[k_index]

                    # divide value by effective number of added elements
            if n > 0:
                replaced_new[k] = replaced_new[k] / n
            else:
                replaced_new[k] = np.nan

        # bulk replace all new values in array
        for k in range(n_nans):
            array[tuple(nan_indices[k])] = replaced_new[k]


        # check if mean square difference between values of replaced
        # elements is below a certain tolerance
        if np.mean((replaced_new - replaced_old) ** 2) < tol:
            break
        else:
                replaced_old = replaced_new

    return array





def get_dist(kernel,kernel_size):

    if len(kernel.shape) == 2:
        # x and y coordinates for each points
        xs, ys = np.indices(kernel.shape)
        # maximal distance form center - distance to center (of each point)
        dist = np.sqrt((ys - kernel_size) ** 2 + (xs - kernel_size) ** 2)
        dist_inv = np.sqrt(2) * kernel_size - dist

    if len(kernel.shape) == 3:
        xs, ys, zs = np.indices(kernel.shape)
        dist = np.sqrt((ys - kernel_size) ** 2 + (xs - kernel_size) ** 2 + (zs - kernel_size) ** 2)
        dist_inv = np.sqrt(3) * kernel_size - dist

    return dist, dist_inv


def replace_outliers(u, v, w=None, method='localmean', max_iter=5, tol=1e-3, kernel_size=1):


    uf = replace_nans(u, method=method, max_iter=max_iter, tol=tol, kernel_size=kernel_size)
    vf = replace_nans(v, method=method, max_iter=max_iter, tol=tol, kernel_size=kernel_size)
    if isinstance(w, np.ndarray):
        wf =  replace_nans(w, method=method, max_iter=max_iter, tol=tol, kernel_size=kernel_size)
        return uf, vf, wf
    return uf, vf


def sig2noise_val(u, v, w=None, sig2noise=None, threshold=1.3):

    ind = sig2noise < threshold

    u[ind] = np.nan
    v[ind] = np.nan
    if isinstance(w,np.ndarray):
        w[ind] = np.nan
        return u, v, w, ind


    return u, v, ind