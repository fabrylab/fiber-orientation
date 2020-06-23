
import numpy as np
import numpy.ma as ma
from numpy.fft import rfft2,irfft2,fftshift
from math import log
from scipy.signal import convolve
import time
import openpiv
from openpiv.process import *
import warnings
from progressbar import *
from pylab import *
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import correlate
from skimage.morphology import disk
from scipy.signal import convolve

def get_field_shape3d(image_size, window_size, overlap):
      n_row = (image_size[0] - window_size[0]) // (window_size[0] - overlap) + 1
      n_col =  (image_size[1] - window_size[1]) // (window_size[1] - overlap) + 1
      n_z = (image_size[2] - window_size[2]) // (window_size[2] - overlap) + 1
      return n_row, n_col, n_z


def plot_3_D(grid, z):
    col = [(0, 0, 1, x / np.max(z)) for x in np.ravel(z)]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(np.ravel(grid[0]), np.ravel(grid[1]), np.ravel(grid[2]), c=col)

    plt.show()


def explode(data):
    # following "https://matplotlib.org/3.1.1/gallery/mplot3d/voxels_numpy_logo.html"

    if len(data.shape) == 3:
        size = np.array(data.shape) * 2
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
    if len(data.shape) == 4: ## color data
        size = np.array(data.shape)[:3] * 2
        data_e = np.zeros(np.concatenate([size - 1,np.array([4])]) , dtype=data.dtype)
        data_e[::2, ::2, ::2,:] = data

    return data_e


def plot_3_D_alpha(data):
    # following "https://matplotlib.org/3.1.1/gallery/mplot3d/voxels_numpy_logo.html"

    col = np.zeros((data.shape[0], data.shape[1], data.shape[2], 4))


    data_fil = data.copy()
    data_fil[(data == Inf)] = np.nanmax(data[~(data == Inf)])
    data_fil =(data_fil - np.nanmin(data_fil)) / (np.nanmax(data_fil) - np.nanmin(data_fil))
    data_fil[np.isnan(data_fil)] = 0

    col[:, :, :, 2] = 1
    col[:, :, :, 3] = data_fil

    col_exp = explode(col)
    fill = explode(np.ones(data.shape))

    x, y, z = np.indices(np.array(fill.shape) + 1).astype(float) // 2

    x[0::2,:,:] += 0.05
    y[:,0::2,:] += 0.05
    z[:,:,0::2] += 0.05
    x[1::2,:,:] += 0.95
    y[:,1::2,:] += 0.95
    z[:,:,1::2] += 0.95


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(x,y,z,fill, facecolors=col_exp, edgecolors=col_exp)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")



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


def simple_defo(x, y):
    f = correlate(x, y, mode="full", method="fft")
    min_pos = np.unravel_index(np.argmax(f), f.shape)
    corr_center = (np.array(f.shape) - 1) / 2
    defo = []
    if len(min_pos) > 0:
        defo.append(corr_center[0] - min_pos[0])
    if len(min_pos) > 1:
        defo.append(corr_center[1] - min_pos[1])  #
    if len(min_pos) > 2:
        defo.append(corr_center[2] - min_pos[2])
    return defo


def extended_search_area_piv3D(frame_a,
                             frame_b,
                             window_size,
                             overlap,
                             dt,
                             search_area_size,
                             drift_correction=False,
                             subpixel_method='gaussian',
                             sig2noise_method=None,
                             width=2,
                             nfftx=None,
                             nffty=None):



    # dsiplacement between window and searcharea can only be uniquely defined if window_size + searcharea_size is an even number
    # this warns, and corrects search_area_size if this is not the case
    search_area_size = check_search_area_size(search_area_size, window_size, warn=True)

    p1 = int(np.ceil((window_size[0] + search_area_size[0]) / 2))
    p2 = int(np.ceil((window_size[1] + search_area_size[1]) / 2))
    p3 = int(np.ceil((window_size[2] + search_area_size[2]) / 2))

    frame_b = np.pad(frame_b,((p1,p1),(p2,p2),(p3,p3))) # padding to
    # get field shape
    n_rows, n_cols, n_z = get_field_shape3d(frame_a.shape, window_size, overlap)

    #    # define arrays
    window_a = np.zeros(window_size, dtype=DTYPEi)
    search_area = np.zeros(search_area_size, dtype=DTYPEi)
    corr = np.zeros(search_area_size, dtype=DTYPEf)
    u = np.zeros([n_rows, n_cols, n_z], dtype=DTYPEf)
    v = np.zeros([n_rows, n_cols, n_z], dtype=DTYPEf)
    w = np.zeros([n_rows, n_cols, n_z], dtype=DTYPEf)
    sig2noise = np.zeros([n_rows, n_cols, n_z], dtype=DTYPEf)

    # loop over the interrogation windows
    # i, j are the row, column indices of the top left corner

    for I,i in tqdm(enumerate(range(0, frame_a.shape[0] - window_size[0], window_size[0] - overlap)), total=n_rows):
        for J,j in enumerate(range(0, frame_a.shape[1] - window_size[1], window_size[1] - overlap)):
            for Z, z in enumerate(range(0, frame_a.shape[2] - window_size[2], window_size[2] - overlap)):
                # get interrogation window matrix from frame a
                window_a = frame_a[i:i+window_size[0], j:j+window_size[1], z:z+window_size[2]]
                #print (i,j,z)
               
                # if  (43 in (np.arange(i,i+window_size[0])) and 43 in np.arange(j,j+window_size[1]) and 43 in np.arange(z,z+window_size[2])):
                #       print(i)

                #       # grid = np.meshgrid((np.arange(i,i+window_size[0])) , np.arange(j,j+window_size[1]) , np.arange(z,z+window_size[2]))
                #       # plot_3_D(grid, window_a)
           


                # get search area using frame b
                r1 = np.array((i - search_area_size[0] / 2, i + search_area_size[0] / 2)) + window_size[0] / 2 + p1 # addinging index shift due to padding
                r2 = np.array((j - search_area_size[1] / 2, j + search_area_size[1] / 2)) + window_size[1] / 2 + p2
                r3 = np.array((z - search_area_size[2] / 2, z + search_area_size[2] / 2)) + window_size[2] / 2 + p3

                r1 = r1.astype(int) # maybe round up
                r2 = r2.astype(int)
                r3 = r3.astype(int)

                search_area = frame_b[r1[0]:r1[1], r2[0]:r2[1], r3[0]:r3[1]]


                


                ##############################
                # compute correlation map
                if (np.sum(window_a!=0)>0) and (np.sum(search_area!=0)>0) :
                    # normalizing_intensity simply substratcs the mean
                    corr = correlate(normalize_intensity(search_area), normalize_intensity(window_a), method="fft", mode="full") # measure time and compare
                    # corr = correlate_windows3D(search_area, window_a,( nfftx=nfftx, nffty=nffty)
                    c = CorrelationFunction3D(corr)
    
                    # find subpixel approximation of the peak center
                    i_peak, j_peak, z_peak = c.subpixel_peak_position(subpixel_method)
    
                    # confirm this// especially if wnidowsize > search area is possibel (should be?)
                    corr_center = (np.array(corr.shape) - 1) / 2
                    v[I, J, Z] = (corr_center[0] - i_peak)/ dt
                    u[I, J, Z] = (corr_center[1]- j_peak)/ dt
                    w[I, J, Z] = (corr_center[2] - z_peak)/ dt

                    if sig2noise_method:
                        sig2noise[I, J, Z] = c.sig2noise_ratio(sig2noise_method, width)
                #plot_3_D_alpha(corr)
                #plot_3_D_alpha(window_a)
                #plot_3_D_alpha(search_area)

    if drift_correction:
           # drift correction
           u= u - np.mean(u)
           v= v - np.mean(v)
           w= w - np.mean(w)

    if sig2noise_method:
        return u, v, w, sig2noise
    else:
        return u, v,w


def correlate_windows3D(window_a, window_b, nfftx=None, nffty=None, nfftz= None):


        if nfftx is None:
            nfftx = 2 * window_a.shape[0]
        if nffty is None:
            nffty = 2 * window_a.shape[1]
        if nfftz is None:
            nfftz = 2 * window_a.shape[2]

        return fftshift(irfftn(rfftn(normalize_intensity(window_a),
                                     s=(nfftx, nffty, nfftz)) *
                               np.conj(rfftn(normalize_intensity(window_b),
                                             s=(nfftx, nffty, nfftz)))).real, axes=(0, 1, 2 ))


class CorrelationFunction3D():
    def __init__(self, corr):
        """A class representing a cross correlation function.

        Parameters
        ----------
        corr : 2d np.ndarray
            the correlation function array

        """
        self.data = corr
        self.shape = self.data.shape

        # get first peak
        self.peak1, self.corr_max1 = self._find_peak(self.data)

    def _find_peak(self, array):
        """Find row and column indices of the highest peak in an array."""

        return  np.unravel_index(np.argmax(array), array.shape), array.max()

    def _find_second_peak(self, width):
        """
        Find the value of the second largest peak.

        The second largest peak is the height of the peak in
        the region outside a ``width * width`` submatrix around
        the first correlation peak.

        Parameters
        ----------
        width : int
            the half size of the region around the first correlation
            peak to ignore for finding the second peak.

        Returns
        -------
        i, j : two elements tuple
            the row, column index of the second correlation peak.

        corr_max2 : int
            the value of the second correlation peak.

        """
        # create a masked view of the self.data array
        tmp = self.data.view(ma.MaskedArray)

        # set width x width square submatrix around the first correlation peak as masked.
        # Before check if we are not too close to the boundaries, otherwise we have negative indices
        r1 = (max(0, self.peak1[0] - width), min(self.peak1[0] + width + 1, self.data.shape[0]))

        r2 = (max(0, self.peak1[1] - width), min(self.peak1[1] + width + 1, self.data.shape[1]))
        r3 = (max(0, self.peak1[2] - width), min(self.peak1[2] + width + 1, self.data.shape[2]))



        tmp[r1[0]:r1[0], r2[0]:r2[0], r3[0]:r3[0]] = ma.masked
        peak, corr_max2 = self._find_peak(tmp)

        return peak, corr_max2

    def subpixel_peak_position(self, method='gaussian'):
        """
        Find subpixel approximation of the correlation peak.

        This function returns a subpixels approximation of the correlation
        peak by using one of the several methods available.

        Parameters
        ----------
        method : string
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

        # the peak and its neighbours: left, right, down, up
        try:
            c =   np.array(self.data[self.peak1[0], self.peak1[1], self.peak1[2]])   # not sure if this is the best way, maybe also include poitns at corners
            c1l = np.array(self.data[self.peak1[0] - 1, self.peak1[1], self.peak1[2]])
            c1r = np.array(self.data[self.peak1[0] + 1, self.peak1[1], self.peak1[2]])
            c2l = np.array(self.data[self.peak1[0], self.peak1[1] - 1, self.peak1[2]])
            c2r = np.array(self.data[self.peak1[0], self.peak1[1] + 1, self.peak1[2]])
            c3l = np.array(self.data[self.peak1[0], self.peak1[1], self.peak1[2] - 1])
            c3r = np.array(self.data[self.peak1[0], self.peak1[1], self.peak1[2] + 1])
        except IndexError:
            # if the peak is near the border do not
            # do subpixel approximation
            return self.peak1

        # if all zero or some is NaN, don't do sub-pixel search:
        tmp = np.array([c, c1l, c1r, c2l, c2r, c3l, c3r])
        if np.any(np.isnan(tmp)) or np.all(tmp == 0):
            return self.peak1

        # replace 0 with small number due to problems with log in gaussian method
        for t in [c, c1l, c1r, c2l, c2r, c3l, c3r]:
            if t == 0 :
                t += 1e-17
   
  
        # if correlation is negative near the peak, fall back
        # to a centroid approximation
        if np.any(tmp < 0) and method == 'gaussian':
            method = 'centroid'

        # choose method
        if method == 'centroid':
            subp_peak_position = (
            ((self.peak1[0] - 1) * c1l + self.peak1[0] * c + (self.peak1[0] + 1) * c1r) / (c1l + c + c1r),
            ((self.peak1[1] - 1) * c2l + self.peak1[1] * c + (self.peak1[1] + 1) * c2r) / (c2l + c + c2r),
            ((self.peak1[2] - 1) * c3l + self.peak1[2] * c + (self.peak1[2] + 1) * c3r) / (c3l + c + c3r))

        elif method == 'gaussian':
            subp_peak_position = (
            self.peak1[0] + ((np.log(c1l) - np.log(c1r)) / (2 * np.log(c1l) - 4 * np.log(c) + 2 * np.log(c1r))),
            self.peak1[1] + ((np.log(c2l) - np.log(c2r)) / (2 * np.log(c2l) - 4 * np.log(c) + 2 * np.log(c2r))),
            self.peak1[2] + ((np.log(c3l) - np.log(c3r)) / (2 * np.log(c3l) - 4 * np.log(c) + 2 * np.log(c3r)))
            )

        elif method == 'parabolic':
            subp_peak_position = (self.peak1[0] + (c1l - c1r) / (2 * c1l - 4 * c + 2 * c1r),
                                  self.peak1[1] + (c2l - c2r) / (2 * c2l - 4 * c + 2 * c2r),
                                  self.peak1[2] + (c3l - c3r) / (2 * c3l - 4 * c + 2 * c3r))
        else:
            raise ValueError("method not understood. Can be 'gaussian', 'centroid', 'parabolic'.")

        return subp_peak_position

    def sig2noise_ratio(self, method='peak2peak', width=2):
        """Computes the signal to noise ratio.

        The signal to noise ratio is computed from the correlation map with
        one of two available method. It is a measure of the quality of the
        matching between two interogation windows.

        Parameters
        ----------
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

        # if the image is lacking particles, totally black it will correlate to very low value, but not zero
        # return zero, since we have no signal.
        if self.corr_max1 < 1e-3:
            return 0.0

        # if the first peak is on the borders, the correlation map is wrong
        # return zero, since we have no signal.
        if (0 in self.peak1 or any(np.array(self.data.shape)-np.array(self.peak1)<=0)):
            return 0.0

        # now compute signal to noise ratio
        if method == 'peak2peak':
            # find second peak height
            peak2, corr_max2 = self._find_second_peak(width=width)

        elif method == 'peak2mean':
            # find mean of the correlation map
            corr_max2 = self.data.mean()

        else:
            raise ValueError('wrong sig2noise_method')

        # avoid dividing by zero
        try:
            sig2noise = self.corr_max1 / corr_max2
        except ValueError:
            sig2noise = np.inf

        return sig2noise


def sig2noise_val_new(u, v, sig2noise, w=None, threshold=1.3):

    mask = sig2noise < threshold

    u[mask] = np.nan
    v[mask] = np.nan
    if isinstance(w,np.ndarray):
        w[mask] = np.nan
        return u, v, w, mask
    return u, v, mask


def replace_nans(array, max_iter, tol, kernel_size = 2, method = 'disk'):

    kernel = np.empty((2 * kernel_size + 1, 2 * kernel_size + 1), dtype=DTYPEf)
    iter_seeds = np.zeros(max_iter, dtype=DTYPEi)

    # indices where array is NaN
    inans, jnans = [x.astype(DTYPEi) for x in np.nonzero(np.isnan(array))]

    # number of NaN elements
    n_nans = len(inans)

    # arrays which contain replaced values to check for convergence
    replaced_new = np.zeros(n_nans, dtype=DTYPEf)
    replaced_old = np.zeros(n_nans, dtype=DTYPEf)

    # depending on kernel type, fill kernel array
    if method == 'localmean':
        for i in range(2 * kernel_size + 1):
            for j in range(2 * kernel_size + 1):
                kernel[i, j] = 1.0

    elif method == 'disk':
        for i in range(2 * kernel_size + 1):
            for j in range(2 * kernel_size + 1):
                if ((kernel_size - i) ** 2 + (kernel_size - j) ** 2) ** 0.5 <= kernel_size:
                    kernel[i, j] = 1.0
                else:
                    kernel[i, j] = 0.0

    elif method == 'distance':
        for i in range(2 * kernel_size + 1):
            for j in range(2 * kernel_size + 1):
                if ((kernel_size - i) ** 2 + (kernel_size - j) ** 2) ** 0.5 <= kernel_size:
                    kernel[i, j] = kernel[i, j] = -1 * (((kernel_size - i) ** 2 + (kernel_size - j) ** 2) ** 0.5 - (
                            (kernel_size) ** 2 + (kernel_size) ** 2) ** 0.5)
                else:
                    kernel[i, j] = 0.0
    else:
        raise ValueError('method not valid. Should be one of `localmean`, `disk` or `distance`.')

    # make several passes
    # until we reach convergence
    for it in range(max_iter):

        # for each NaN element
        for k in range(n_nans):
            i = inans[k]
            j = jnans[k]

            # init to 0.0
            replaced_new[k] = 0.0
            n = 0.0

            # loop over the kernel
            for I in range(2 * kernel_size + 1):
                for J in range(2 * kernel_size + 1):

                    # if we are not out of the boundaries
                    if i + I - kernel_size < array.shape[0] and i + I - kernel_size >= 0:
                        if j + J - kernel_size < array.shape[1] and j + J - kernel_size >= 0:

                            # if the neighbour element is not NaN itself.
                            if not np.isnan(array[i + I - kernel_size, j + J - kernel_size]):

                                # do not bother with 0 kernel values
                                if kernel[I, J] != 0:
                                    # convolve kernel with original array
                                    replaced_new[k] = replaced_new[k] + array[
                                        i + I - kernel_size, j + J - kernel_size] * \
                                                      kernel[I, J]
                                    n = n + kernel[I, J]

            # divide value by effective number of added elements
            if n > 0:
                replaced_new[k] = replaced_new[k] / n
            else:
                replaced_new[k] = np.nan

        # bulk replace all new values in array
        for k in range(n_nans):
            array[inans[k], jnans[k]] = replaced_new[k]

        # check if mean square difference between values of replaced
        # elements is below a certain tolerance
        if np.mean((replaced_new - replaced_old) ** 2) < tol:
            break
        else:
            for l in range(n_nans):
                replaced_old[l] = replaced_new[l]

    return array




def replace_nans_new(a, max_iter, tol, kernel_size = 2, method = 'disk'):


    # new method:
    # convloulution is (in this setting) defined as c = sum(wi * i), with sum(wi) = 1
    #  i is pixel intensity ans wi is the weight at a position defined by the convolution kernel
    #  if we want to  ignore nans in the convolution window we have to correct wi since some positions are no longer summed
    # the correction is deviding by the sum of all weights that at not at nan position
    # wi_n=wi/sum(wi*j) j=0 if i is nan else 1 // this means that sum(wi_n) for i not nan = 1


    array = a.copy()
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

    nan_mask = np.isnan(array)
    nan_mas_init = nan_mask.copy()
    replaced_old = np.zeros(np.sum(nan_mask))# only nans in the begining

    for it in range(max_iter):

        # breaking if nothing to replace
        if np.sum(nan_mask) == 0:
            break

        # replacing none with convolution
        non_nans = (~nan_mask).astype(int)
        ar_zero_fill = array.copy()
        ar_zero_fill[nan_mask] = 0
        conv = convolve(ar_zero_fill, kernel, mode="same")/ convolve(non_nans, kernel, mode="same") # works in 3d in 2d
        array[nan_mask] = conv[nan_mask]

        # checking tolerance  ## but what do we need this for
        replaced_new = array[nan_mas_init]
        replaced_new[np.isnan(replaced_new)] = 0
        if np.mean((replaced_new - replaced_old) ** 2) < tol:
            break

        # reassining nan region and replaced list
        replaced_old = replaced_new
        nan_mask = np.isnan(array)

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


def replace_outliers_new(u, v, w=None, method='localmean', max_iter=5, tol=1e-3, kernel_size=1):


    uf = replace_nans(u, method=method, max_iter=max_iter, tol=tol, kernel_size=kernel_size)
    vf = replace_nans(v, method=method, max_iter=max_iter, tol=tol, kernel_size=kernel_size)
    if np.isinstance(w, np.ndarray):
        wf = w.copy()
        return uf, vf, wf
    return uf, vf






"""
load stacks
"""

# out_folder = r"\\131.188.117.96\biophysDS\dboehringer\Platte_4\Software\3d-openpiv\test"
# # making a 3d array from a stack
# folder1=r"\\131.188.117.96\biophysDS/dboehringer/20170914_A172_rep1-pos01/After/"
# images=[os.path.join(folder1,x) for x in os.listdir(folder1) if "_ch00.tif" in x]
# im_shape = plt.imread(images[0]).shape
# stack1 = np.zeros((im_shape[0],im_shape[1],len(images)))

# for i,im in enumerate(images):
#     stack1[:,:,i] = np.mean(plt.imread(im),axis=2)

# folder1=r"\\131.188.117.96\biophysDS/dboehringer/20170914_A172_rep1-pos01/Before/"
# images=[os.path.join(folder1,x) for x in os.listdir(folder1) if "_ch00.tif" in x]
# im_shape = plt.imread(images[0]).shape
# stack2 = np.zeros((im_shape[0],im_shape[1],len(images)))

# for i,im in enumerate(images):
#     stack2[:,:,i] = np.mean(plt.imread(im),axis=2)


"""
test stacks
"""
out_folder = r"\\131.188.117.96\biophysDS\dboehringer\Platte_4\Software\3d-openpiv\test-system"


#  test single defo
# sphere1 = np.zeros((10,10,10))
# sphere1[4,4,1] = 1

# sphere2 =  np.zeros((10,10,10))
# sphere2[4,3,2] = 1

#  test cube defo
sphere1 = np.zeros((10,10,10))
sphere1[5,5,3:7] = 1

sphere2 =  np.zeros((10,10,10))
sphere2[5,5,3:8] = 1


# window_size = (6,6,6)
# overlap = 3  #11                # piv issue even-odd numbers.. ?
# search_area =  (6,6,6)


# test sphere defo
# center = (50, 50, 50)
# size = (100, 100, 100)
# distance = np.linalg.norm(np.subtract(np.indices(size).T,np.asarray(center)), axis=len(center))
# sphere1 = np.ones(size) * (distance<=30)

# center = (50, 50, 50)
# size = (100, 100, 100)
# distance = np.linalg.norm(np.subtract(np.indices(size).T,np.asarray(center)), axis=len(center))
# sphere2 = np.ones(size) * (distance<=15)

#Plot sphere
# fig =plt.figure(figsize=(6,6))
# ax = fig.gca(projection='3d')
# ax.voxels(sphere2, facecolors="r", edgecolor='k',alpha=0.4)
# ax.voxels(sphere1, facecolors="b", edgecolor='k',alpha=0.4)
# plt.show()

#
# ax = fig.gca(projection='3d')
# ax.voxels(sphere2, facecolors="r", edgecolor='k',alpha=0.4)
# ax.voxels(sphere1, facecolors="b", edgecolor='k',alpha=0.4)
# plt.show()





# 3d piv
#  test cube defo
sphere1 = np.zeros((10,10,10))
sphere1[5,5,5:6] = 1

sphere2 =  np.zeros((10,10,10))
sphere2[5,5,5:7] = 1



center = (8, 8, 8)
size = (16, 16, 16)
distance = np.linalg.norm(np.subtract(np.indices(size).T,np.asarray(center)), axis=len(center))
sphere1 = np.ones(size) * (distance<=4)


center = (8, 8, 8)
size = (16, 16, 16)
distance = np.linalg.norm(np.subtract(np.indices(size).T,np.asarray(center)), axis=len(center))
sphere2 = np.ones(size) * (distance<=5)


window_size = (5,5,5)
overlap = 4#11
search_area = (5,5,5)

n_rows, n_cols, n_z = get_field_shape3d(sphere1.shape, window_size, overlap)
print("needs %s iterations"%str(n_rows))

u, v, w, sig2noise = extended_search_area_piv3D(sphere1, sphere2, window_size, overlap, 1, search_area, subpixel_method='gaussian',
                             sig2noise_method='peak2peak',
                             width=2,
                             nfftx=None,
                             nffty=None)


#np.save(os.path.join(out_folder,"u.npy"), u)
#np.save(os.path.join(out_folder,"v.npy"), v)
#np.save(os.path.join(out_folder,"w.npy"), w)
#np.save(os.path.join(out_folder,"sig_noise.npy"), sig2noise)







"""
visualize results
"""

# # add color
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from itertools import chain
# make grid
x, y, z = np.indices(u.shape)
distance = np.sqrt(x**2+y**2+z**2)
deformation = np.sqrt(u**2+v**2+w**2)

#cbound=[0,10]
#filter defos - or use 0 100
mask_filtered = (np.sqrt(u**2+v**2+w**2)>=np.nanpercentile(np.sqrt(u**2+v**2+w**2),0)) &(np.sqrt(u**2+v**2+w**2)<=np.nanpercentile(np.sqrt(u**2+v**2+w**2),100))
mask_filtered = deformation>0.5
mask_filtered = np.ones(x.shape).astype(bool)

offset = np.random.uniform(0,0.2,x[mask_filtered].shape)
xf = x[mask_filtered]
yf = y[mask_filtered]
zf = z[mask_filtered]
uf = u[mask_filtered]
vf = v[mask_filtered]
wf = w[mask_filtered]
df = deformation[mask_filtered]

# make cmap
cbound=[0, np.nanmax(df)]
# create normalized color map for arrows
norm = matplotlib.colors.Normalize(vmin=cbound[0],vmax=cbound[1])
sm = matplotlib.cm.ScalarMappable(cmap="jet", norm=norm)
sm.set_array([])
# different option
#colors = sm.to_rgba(np.ravel(deformation))
colors = matplotlib.cm.jet(norm(df)) #
# plot the data
#colors[deformation==1.5]=np.array([1,1,1,1])

colors = [c for c,d in zip(colors, df) if d > 0] + list(chain(*[[c, c] for c,d in zip(colors, df) if d > 0]))
# colors in ax.quiver 3d is really fucked up/ will probabaly change with updates:
# requires list with: first len(u) entries define the colors of the shaft, then the next len(u)*2 entries define
# the color of alternating left and right arrow head side. Try for example:
# colors = ["red" for i in range(len(cf))] + list(chain(*[["blue", "yellow"] for i in range(len(cf))]))
# to see this effect
# BUT WAIT THERS MORE: zeor length arrows are apparently filtered out in the matplolib with out filtering the color list appropriately
# so we have to do this our selfes as well



fig = plt.figure()
ax = fig.gca(projection='3d', rasterized=True)

quiver_filtered = ax.quiver(xf, yf, zf, uf, vf, wf ,
                          colors=colors,  normalize=True, alpha =0.9, arrow_length_ratio=0,  pivot='tip', linewidth=0.5)
plt.colorbar(sm)
ax.set_xlim(x.min(),x.max())
ax.set_ylim(y.min(),y.max())
ax.set_zlim(z.min(),z.max())
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.w_xaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
ax.w_yaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
ax.w_zaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))

#plot_3_D_alpha(sphere1)
#plot_3_D_alpha(sphere2)

s2=sig2noise.copy()
s2[sig2noise==1]=0
#plot_3_D_alpha(s2)



# signal to noise filtering and stuff # not tested yet
u, v, w, mask = sig2noise_val_new( u, v, sig2noise, w=w, threshold=1.05)
u, v, w = replace_outliers_new(u, v, w=w, method='localmean', max_iter=10, kernel_size=2)
