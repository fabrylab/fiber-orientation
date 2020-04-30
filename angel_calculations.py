import numpy as np
import copy
from tqdm import tqdm
from utilities import *


def generate_rotation_matrix(angle):
    return np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])

def rotate_track(ps, vecs, angle):
    com = np.nanmean(ps,axis=0)
    # centralize points
    pc = ps-com
    # apply rotation
    pc_rot = np.matmul(pc, generate_rotation_matrix(angle).T)
    vecs_rot = np.matmul(vecs, generate_rotation_matrix(angle).T) # applie same rotation to vectos, that works ?
    p = pc_rot + com # undo centralization
    return p, vecs_rot


def get_mean_anlge_over_time(frames, min_frame, max_frame, binsize_time, step_size, angs):
    [a] = binning_by_frame(frames, min_frame, max_frame, binsize_time, step_size, angs)
    ma = [np.nanmean(angs) for angs in a.values()]
    return ma

class FilterAndWeighting():

    def __init__(self, angles, lengths, points, frames):
        self.dtype=None
        if all([isinstance(x,dict) for x in [angles,lengths,points,frames]]):
            self.dtype=dict
        if all([isinstance(x,np.ndarray) for x in [angles,lengths,points, frames]]):
            self.dtype=np.ndarray
        if self.dtype is None:
            raise TypeError("Input must either be all dictionaryies or all np.ndarrays")

        self._angles = copy.deepcopy(angles)
        self._points = copy.deepcopy(points)
        self._lengths = copy.deepcopy(lengths)
        self._frames = copy.deepcopy(frames)
        if self.dtype==dict: # construct a refernece list of keys
            self.keys=[]
            for key, values in self._angles.items():
                self.keys.extend([key]*len(values))
            self.keys = np.array(self.keys)
            self.unique_keys=np.unique(self.keys)
            # convert to arrays
            self._angles=np.hstack(list(self._angles.values()))
            self._points = np.vstack(list(self._points.values()))
            self._lengths = np.hstack(list(self._lengths.values()))
            self._frames = np.hstack(list(self._frames.values()))

    @property
    def angles(self):
        # reconstructing dictionary
        if self.dtype==dict:
            return {key:self._angles[self.keys==key] for key in self.unique_keys}
        else:
            return self._angles

    @property
    def points(self):
        if self.dtype==dict:
            return {key:self._points[self.keys==key] for key in self.unique_keys}
        else:
            return self._points
    @property
    def lengths(self):
        if self.dtype==dict:
            return {key:self._lengths[self.keys==key] for key in self.unique_keys}
        else:
            return self._lengths
    @property
    def frames(self):
        if self.dtype==dict:
            return {key:self._frames[self.keys==key] for key in self.unique_keys}
        else:
            return self._frames




    def apply_filter(self, filter_list):
        # filter list is list of  tupels with [(function ( bound method from the class instnace), {parapeters}),]

        fmask=np.ones(len(self._angles))
        for filter in filter_list:
            fmask *= filter[0](self,**filter[1]) # logical and connection
        fmask = fmask.astype(bool)
        self._angles = self._angles[fmask]
        self._points = self._points[fmask]
        self._lengths = self._lengths[fmask]
        self._frames = self._frames[fmask]
        self.keys = self.keys[fmask]

    def apply_weighting(self, weighting_list):
        # weighting_list is list of  tupels with [(function ( bound method from the class instnace), {parapeters}),]
        for weight in weighting_list:
            weight[0](self,**weight[1])


    # weighting functions
    def linear_weigthing(self):
        self._angles=self._angles*self._lengths/np.nanmean(self._lengths)


    # filtering functions
    def length_threshold(self, threshold=1):
        filter=self._lengths > threshold
        return filter

    def spatial_filter_radius(self, center=(0,0), radius=1):
        dist_to_center = np.linalg.norm(self._points - center,axis=1)
        filter = dist_to_center > radius
        return filter

    def spatial_filter_area(self, mask, inside_outside="inside"):
        mask = mask.astype(bool)
        # should use an area
        #if inside_outside=="inside":
        # filter=mask[self.points[:,1].astype(int),self.points[:,1].astype(int)] #### ####### check if x and y is correct
        #if inside_outside=="ouside":
            # filter=~mask[self.points[:,1].astype(int),self.points[:,1].astype(int)]
        # return filter


        pass





def weight_by_length(angles, lengths, method="lw", threshold=1):
    if method == "lw":
        angles_weighted = angles * lengths / np.sum(lengths)
    if method == "tw":
        angles_weighted = copy.deepcopy(angles)
        angles_weighted[lengths < threshold] = np.nan
    if method == "nw":
        angles_weighted = copy.deepcopy(angles)

    return angles_weighted





def radial_angle_distribution(points,center,angles,bins):
    rs = np.linalg.norm(points - center, axis=1)
    h1, bins1 = np.histogram(rs, bins=bins, weights=angles)
    h2, bins2 = np.histogram(rs, bins=bins)
    hist = h1 / h2
    bins = (bins1[:-1] + bins1[1:]) / 2
    return bins, hist

def spatial_angle_distribution(points,angles,bins=None):
    if not isinstance(bins,(np.ndarray, int, tuple)):
        bins = 10 # defuatl value innp.histogram2d
    hist1, bins1x, bins1y = np.histogram2d(points[:,0], points[:,1], bins=bins, weights=angles)
    hist2, bins2x, bins2y = np.histogram2d(points[:, 0], points[:, 1], bins=bins)
    hist = np.swapaxes(hist1/hist2, axis1=0, axis2=1) # hist returns x,y array, im is in y,x array
    return hist

def project_angle(a):
    # projects angles in range [0,2pi] to [0,pi/2] to extract measure of parallelicity
    angle=copy.deepcopy(a)
    if isinstance(angle,(float,int)):
        angle = np.array([angle])
    if not isinstance(angle,np.ndarray):
        angle = np.array(angle)

     # this is mathematically equivalent to (np.pi/2)-np.abs((angle%np.pi)-(np.pi/2))
    angle[angle > np.pi ] = 2 * np.pi - angle[angle > np.pi ] # mapping all values > np.pi
    angle[angle > np.pi/2] = np.pi- angle[angle > np.pi/2 ]   # mapping all values > np.pi/2
    return angle


def calculate_angle(vec1, vec2, axis=0):
    v1 = np.expand_dims(vec1, axis=0) if len(np.array(vec1).shape) == 1 else np.array(vec1)
    v2 = np.expand_dims(vec2, axis=0) if len(np.array(vec2).shape) == 1 else np.array(vec2)

    # angles towards x axis in range [pi,-pi]
    angles1 = np.arctan2(np.take(v1, 1, axis), np.take(v1, 0, axis)) # xycoordinates are reversed
    angles2 = np.arctan2(np.take(v2, 1, axis), np.take(v2, 0, axis))# xycoordinates are reversed
    # angle between vector in range [0,2pi]
    angle = (angles1-angles2) % (2 * np.pi)  # not sure how this works exactely
    # taking the modulo should be the same as +2pi if angle <0 which should  be correct......
    return angle


def binning_by_frame(frame_list, min_frame, max_frame, binsize_time, step_size, *args):
    ## returns dictinoary of binned values (binned by frames list) for all lists in args
    dict_list = [{} for i in range(len(args))]
    for i, s in enumerate(range(min_frame, max_frame - binsize_time, step_size)):
         time_mask = np.logical_and(frame_list >= s, frame_list < (s + binsize_time))
         for i, a in enumerate(args):
             dict_list[i][(s,s + binsize_time)] = a[time_mask]
    return dict_list



def extract_angles_area(hist_mask, points_b, *args, dtype="array"):

    if dtype==dict:
        hist_mask = hist_mask.astype(bool)
        points_select = {}
        others = [{} for i in range(len(args))]
        for f_range in tqdm(points_b.keys()):
            points_bin = points_b[f_range]
            inside_select = hist_mask[np.round(points_bin[:, 1]).astype(int), np.round(points_bin[:, 0]).astype(int)]
            for i, ar in enumerate(args):
                others[i][f_range] = ar[f_range][inside_select]
    if dtype=="array":
        hist_mask = hist_mask.astype(bool)
        inside_select = hist_mask[np.round(points_b[:, 1]).astype(int), np.round(points_b[:, 0]).astype(int)]
        points_select = points_b[inside_select]
        others = []
        for i, ar in enumerate(args):
            others.append(ar[inside_select])
    return points_select, others

