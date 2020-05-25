
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




def get_field_shape3d(image_size, window_size, overlap):
      n_row = (image_size[0] - window_size[0]) // (window_size[0] - overlap) + 1
      n_col =  (image_size[1] - window_size[1]) // (window_size[1] - overlap) + 1
      n_z = (image_size[2] - window_size[2]) // (window_size[2] - overlap) + 1
      return n_row, n_col, n_z


def extended_search_area_piv3D(frame_a,
                             frame_b,
                             window_size,
                             overlap,
                             dt,
                             search_area_size,
                             subpixel_method='gaussian',
                             sig2noise_method=None,
                             width=2,
                             nfftx=None,
                             nffty=None):

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



                # get search area using frame b
                r1 = np.array((i - search_area_size[0] / 2, i + search_area_size[0] / 2)) + window_size[0] / 2 + p1 # addinging index shift due to padding
                r2 = np.array((j - search_area_size[1] / 2, j + search_area_size[1] / 2)) + window_size[1] / 2 + p2
                r3 = np.array((z - search_area_size[2] / 2, z + search_area_size[2] / 2)) + window_size[2] / 2 + p3

                r1 = r1.astype(int) # maybe round up
                r2 = r2.astype(int)
                r3 = r3.astype(int)

                search_area = frame_b[r1[0]:r1[1], r2[0]:r2[1], r3[0]:r3[1]]


                # compute correlation map
                if np.sum(window_a!=0)>0:
                    corr = correlate_windows3D(search_area, window_a, nfftx=nfftx, nffty=nffty)
                    c = CorrelationFunction3D(corr)

                    # find subpixel approximation of the peak center
                    i_peak, j_peak, z_peak = c.subpixel_peak_position(subpixel_method)

                    # velocities ##### rethink this plz
                    v[I, J, Z] = -((i_peak - corr.shape[0] / 2) - (search_area_size[0] - window_size[0]) / 2) / dt
                    u[I, J, Z] = ((j_peak - corr.shape[1] / 2) - (search_area_size[1] - window_size[1]) / 2) / dt
                    w[I, J, Z] = ((z_peak - corr.shape[2] / 2) - (search_area_size[2] - window_size[2]) / 2) / dt

                    # compute signal to noise ratio
                    if sig2noise_method:
                        sig2noise[I, J, Z] = c.sig2noise_ratio(sig2noise_method, width)
                else:
                    v[I, J] = 0.0
                    u[I, J] = 0.0
                    # compute signal to noise ratio
                    if sig2noise_method:
                        sig2noise[I, J] = np.inf



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
            c = self.data[self.peak1[0], self.peak1[1], self.peak1[2]]   # not sure if this is the best way, maybe also include poitns at corners
            c1l = self.data[self.peak1[0] - 1, self.peak1[1], self.peak1[2]]
            c1r = self.data[self.peak1[0] + 1, self.peak1[1], self.peak1[2]]
            c2l = self.data[self.peak1[0], self.peak1[1] - 1, self.peak1[2]]
            c2r = self.data[self.peak1[0], self.peak1[1] + 1, self.peak1[2]]
            c3l = self.data[self.peak1[0], self.peak1[1], self.peak1[2] - 1]
            c3r = self.data[self.peak1[0], self.peak1[1], self.peak1[2] + 1]
        except IndexError:
            # if the peak is near the border do not
            # do subpixel approximation
            return self.peak1

        # if all zero or some is NaN, don't do sub-pixel search:
        tmp = np.array([c, c1l, c1r, c2l, c2r, c3l, c3r])
        if np.any(np.isnan(tmp)) or np.all(tmp == 0):
            return self.peak1

        # if correlation is negative near the peak, fall back
        # to a centroid approximation
        if np.any(tmp < 0) and method == 'gaussian':
            method = 'centroid'

        # choose method
        if method == 'centroid':
            subp_peak_position = (
            ((self.peak1[0] - 1) * c1l + self.peak1[0] * c + (self.peak1[0] + 1) * c1r) / (c1l + c + c1r),
            ((self.peak1[1] - 1) * c2l + self.peak1[1] * c + (self.peak1[1] + 1) * c2r) / (c2l + c + c2r),
            ((self.peak1[1] - 1) * c3l + self.peak1[1] * c + (self.peak1[1] + 1) * c3r) / (c3l + c + c3r))

        elif method == 'gaussian':
            subp_peak_position = (
            self.peak1[0] + ((np.log(c1l) - np.log(c1r)) / (2 * np.log(c1l) - 4 * np.log(c) + 2 * np.log(c1r))),
            self.peak1[1] + ((np.log(c2l) - np.log(c2r)) / (2 * np.log(c2l) - 4 * np.log(c) + 2 * np.log(c2r))),
            self.peak1[1] + ((np.log(c3l) - np.log(c3r)) / (2 * np.log(c3l) - 4 * np.log(c) + 2 * np.log(c3r)))
            )

        elif method == 'parabolic':
            subp_peak_position = (self.peak1[0] + (c1l - c1r) / (2 * c1l - 4 * c + 2 * c1r),
                                  self.peak1[0] + (c2l - c2r) / (2 * c2l - 4 * c + 2 * c2r),
                                  self.peak1[0] + (c3l - c3r) / (2 * c3l - 4 * c + 2 * c3r))
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

def extended_search_area_piv2D(frame_a,
                             frame_b,
                             window_size,
                             overlap,
                             dt,
                             search_area_size,
                             subpixel_method='gaussian',
                             sig2noise_method=None,
                             width=2,
                             nfftx=None,
                             nffty=None):


    # get field shape
    n_rows, n_cols = get_field_shape((frame_a.shape[0], frame_a.shape[1]), window_size, overlap)

    #    # define arrays
    window_a = np.zeros([window_size, window_size], dtype=DTYPEi)
    search_area = np.zeros([search_area_size, search_area_size], dtype=DTYPEi)
    corr = np.zeros([search_area_size, search_area_size], dtype=DTYPEf)
    u = np.zeros([n_rows, n_cols], dtype=DTYPEf)
    v = np.zeros([n_rows, n_cols], dtype=DTYPEf)
    sig2noise = np.zeros([n_rows, n_cols], dtype=DTYPEf)

    # loop over the interrogation windows
    # i, j are the row, column indices of the top left corner
    I = 0
    for i in range(0, frame_a.shape[0] - window_size, window_size - overlap):
        J = 0
        for j in range(0, frame_a.shape[1] - window_size, window_size - overlap):

            # get interrogation window matrix from frame a
            for k in range(window_size):
                for l in range(window_size):
                    window_a[k, l] = frame_a[i + k, j + l]

            # get search area using frame b
            for k in range(search_area_size):
                for l in range(search_area_size):

                    # fill with zeros if we are out of the borders
                    if i + window_size / 2 - search_area_size / 2 + k < 0 or i + window_size / 2 - search_area_size / 2 + k >= \
                            frame_b.shape[0]:
                        search_area[k, l] = 0
                    elif j + window_size / 2 - search_area_size / 2 + l < 0 or j + window_size / 2 - search_area_size / 2 + l >= \
                            frame_b.shape[1]:
                        search_area[k, l] = 0
                    else:
                        search_area[k, l] = frame_b[
                            int(i + window_size / 2 - search_area_size / 2 + k), int(j + window_size / 2 - search_area_size / 2 + l)]

            imshow(window_a, cmap=cm.gray)
            show()
            imshow(search_area, cmap=cm.gray)
            show()

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

"""
load stacks
"""

out_folder = r"\\131.188.117.96\biophysDS\dboehringer\Platte_4\Software\3d-openpiv\test"
# making a 3d array from a stack
folder1=r"\\131.188.117.96\biophysDS/dboehringer/20170914_A172_rep1-pos01/After/"
images=[os.path.join(folder1,x) for x in os.listdir(folder1) if "_ch00.tif" in x]
im_shape = plt.imread(images[0]).shape
stack1 = np.zeros((im_shape[0],im_shape[1],len(images)))

for i,im in enumerate(images):
    stack1[:,:,i] = np.mean(plt.imread(im),axis=2)

folder1=r"\\131.188.117.96\biophysDS/dboehringer/20170914_A172_rep1-pos01/Before/"
images=[os.path.join(folder1,x) for x in os.listdir(folder1) if "_ch00.tif" in x]
im_shape = plt.imread(images[0]).shape
stack2 = np.zeros((im_shape[0],im_shape[1],len(images)))

for i,im in enumerate(images):
    stack2[:,:,i] = np.mean(plt.imread(im),axis=2)
"""
3d piv
"""

window_size = (51,51,51)
overlap = 31  #11                # piv issue even-odd numbers.. ?
search_area =  (51,51,51)
n_rows, n_cols, n_z = get_field_shape3d(stack1.shape, window_size, overlap)
print("needs %s iterations"%str(n_rows))

u, v, w, sig2noise = extended_search_area_piv3D(stack1, stack2, window_size, overlap, 1, search_area, subpixel_method='gaussian',
                             sig2noise_method='peak2peak',
                             width=2,
                             nfftx=None,
                             nffty=None)


np.save(os.path.join(out_folder,"u.npy"), u)
np.save(os.path.join(out_folder,"v.npy"), v)
np.save(os.path.join(out_folder,"w.npy"), w)
np.save(os.path.join(out_folder,"sig_noise.npy"), sig2noise)

"""
visualize results
"""

# # add color
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d', rasterized=True)
# make grid
x_ = np.arange(0., n_rows,1)
y_ = np.arange(0., n_cols,1)
z_ = np.arange(0., n_z,1)
x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
#filter defos - or use 0 100
mask_filtered = (np.sqrt(u**2+v**2+w**2)>=np.nanpercentile(np.sqrt(u**2+v**2+w**2),0)) &(np.sqrt(u**2+v**2+w**2)<=np.nanpercentile(np.sqrt(u**2+v**2+w**2),100)) 
# make cmap
distance =np.sqrt(x**2+y**2+z**2)
deformation = np.sqrt(u**2+v**2+w**2)[mask_filtered]
#cbound=[0,10]  
cbound=[0,np.max(deformation)]
# create normalized color map for arrows
norm = matplotlib.colors.Normalize(vmin=cbound[0],vmax=cbound[1])
cm = matplotlib.cm.jet
sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])
# different option 
#colors = sm.to_rgba(np.ravel(deformation))
#colors = matplotlib.cm.jet( (deformation-cbound[0])/(cbound[1]-cbound[0]) ) # 
# plot the data
quiver_filtered = ax.quiver(x[mask_filtered], y[mask_filtered], z[mask_filtered], u[mask_filtered], v[mask_filtered], w[mask_filtered]  ,
                          normalize=True ,alpha=0.7,  pivot='tip', cmap=cm, norm=norm      )  # c ,length=1.3, arrow_length_ratio=1,  linewidth=0.5
ax.w_xaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
ax.w_yaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
ax.w_zaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
plt.colorbar(sm)
ax.set_xlim(x.min(),x.max())
ax.set_ylim(y.min(),y.max())
ax.set_zlim(z.min(),z.max())


