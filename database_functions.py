# functions handling accessing and writing to clickpoints
import clickpoints
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict
import sys
sys.path.append("/home/user/Software/fiber-orientation")
import traceback
from utilities import *
from angles import rotate_track

class OpenDB:
    # context manager for database file. if db is a path a new file handle is opened
    # this handle is later closed (!). if db is already clickpoints.DataFile object,
    # the handle is not closed
    def __init__(self,db, method="r", raise_Error=True):
        self.raise_Error = raise_Error
        if isinstance(db, clickpoints.DataFile):
            self.file = db
            self.db_obj=True
        else:
            self.file = clickpoints.DataFile(db, method)
            self.db_obj = False

    def __enter__(self):
        return self.file

    def __exit__(self, exc_type, exc_value, trace):

        if not self.db_obj:
            self.file.db.close()
        if self.raise_Error:
            return False
        else:
            traceback.print_tb(trace)
            return True



def remove_empty_entries(dicts, dtype="list"):
     if dtype=="list":
         for d in dicts:
             empty_key = [key for key, value in d.items() if len(value) == 0]
             for ek in empty_key:
                 d.pop(ek)

def make_iterable(value):
    if not hasattr(value, '__iter__') or isinstance(value,str):
        return [value]
    else:
        return value


def read_tracks_iteratively(db, track_types=None, start_frame=0, end_frame=None, nanfill=False):
    # track_type=None for all tracks
    tracks_dict = defaultdict(list)  ### improve by using direct sql query or by iterating through frames
    if not track_types is None:
        track_types = make_iterable(track_types)

    with OpenDB(db) as db_l:

        if end_frame is None:
            frames = list(range(start_frame, db_l.getImageCount()))
        else:
            frames = list(range(start_frame, end_frame))

        for i, m in enumerate(tqdm(db_l.getMarkers(frame=frames,type=track_types))):
            if not m.track_id is None: # this excludes all non-track markers
                tracks_dict[m.track_id].append([m.x, m.y, m.image.sort_index])

    tracks_dict = {k: np.array(v) for k,v in tracks_dict.items() if len(v)>0} # not necessary??
    if nanfill:
        ret_tracks_dict = {}
        for k,v in tracks_dict.items():
            f_range = (int(np.min(v[:,2])),int(np.max(v[:,2]))) # range of frames
            n_array = np.zeros((1+f_range[1]-f_range[0],3)) + np.nan # nan filled array of appropriate length
            n_array[v[:,2].astype(int) - f_range[0]] = v # filling at the correct positions
            ret_tracks_dict[k]=n_array
        return ret_tracks_dict
    return tracks_dict

def read_tracks_NanPadded(db, start_frame=0, end_frame=None, track_types=None):
    with OpenDB(db) as db_l:
        if not isinstance(end_frame, int):
            end_frame = db_l.getImageCount()
        tracks = db_l.getTracksNanPadded(start_frame=start_frame, end_frame=end_frame, track_types=track_types)
    return tracks

def randomize_tracks(vecs, points, frames, im_shape):
    vecs_rot = {}
    points_rot = {}
    frames_rot = {}
    for (k_vec, v_vec), (k_ps, v_ps) in zip(vecs.items(), points.items()):
        ps_rot, vs_rot = rotate_track(v_ps, v_vec, np.random.uniform(0, np.pi * 2))
        inside_mask = ((ps_rot[:,0]<im_shape[1]) * (ps_rot[:,1]<im_shape[0]) * (ps_rot[:,0]>0) * (ps_rot[:,1]>0)).astype(bool)# points are in xy coordinates
        vecs_rot[k_vec] = vs_rot[inside_mask]
        points_rot[k_ps] = ps_rot[inside_mask]
        frames_rot[k_ps] = frames[k_ps][inside_mask]

    return  vecs_rot, points_rot, frames_rot



def read_tracks_list_by_frame(db, window_size=1, start_frame=0, end_frame=None, return_dict=False, track_types=None):

    tracks_dict = read_tracks_iteratively(db, track_types=track_types, start_frame=start_frame, end_frame=end_frame,
                                          nanfill=True)

    vecs = {k: v[window_size:, [0, 1]] - v[:-window_size, [0, 1]] for k, v in tracks_dict.items()}
    points = {k: v[window_size:, [0, 1]] for k, v in tracks_dict.items()}
    frames = {k: v[window_size:, 2] for k, v in tracks_dict.items()}
    nan_filter = {k: ~np.isnan(v) for k, v in vecs.items()}

    vecs = {k: v[nan_filter[k][:, 0]] for k, v in vecs.items()}
    points = {k: v[nan_filter[k][:, 0]] for k, v in points.items()}
    frames = {k: v[nan_filter[k][:, 0]] for k, v in frames.items()}
    remove_empty_entries([vecs,points,frames], dtype="list")
    if return_dict:
        return vecs, points, frames
    else:
        return np.vstack(list(vecs.values())), np.vstack(list(points.values())), np.hstack(list(frames.values()))





def get_orientation_line(db):
    with OpenDB(db) as db_l:
        e_points = [(m.x, m.y) for m in db_l.getMarkers(type="elips")]
        straight_line = np.array(e_points[0]) - np.array(e_points[3])

    return straight_line


def read_tracks_list_by_id(db, min_id=0, max_id=np.inf):
    #db path or db object
    with OpenDB(db, mehtod="r"):
        all_vecs = []
        all_points = []
        all_frames = []
        track_iter=db.getTracks()
        max_id = len(track_iter) if len(track_iter) < max_id else max_id
        for i,t in tqdm(enumerate(track_iter),total=max_id):
            if i>=min_id and i<max_id:
                ps = t.points
                vecs = ps[1:] - ps[:-1] # vector between individual detections
                all_vecs.append(vecs)
                all_points.append(ps[:-1])   # origin of the vector
                all_frames.append(t.frames[:-1]) # all time points
        all_vecs = np.concatenate(all_vecs,axis=0)
        all_points = np.concatenate(all_points,axis=0)
        all_frames = np.concatenate(all_frames,axis=0)

    return all_vecs, all_points, all_frames


