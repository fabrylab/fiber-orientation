# functions handling accessing and writing to clickpoints
import clickpoints
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict
import sys
sys.path.append("/home/user/Software/fiber-orientation")
import traceback


def createFolder(directory):
    '''
    function to create directories if they dont already exist
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    except OSError:
        print('Error: Creating directory. ' + directory)

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




def read_tracks_list_by_frame(db, window_size=1, start_frame=0, end_frame=None, return_dict=False):

    with OpenDB(db) as db_l:
        if not isinstance(end_frame,int):
            end_frame = db_l.getImageCount()
        tracks = db_l.getTracksNanPadded(start_frame=start_frame, end_frame=end_frame)
         # calcualting vectors
        n_vecs = tracks.shape[0] * (tracks.shape[1] - window_size)
        vecs = tracks[:, window_size:, :] - tracks[:, :-window_size, :]
        points = tracks[:, :-window_size, :]
        if return_dict:
            nan_dict = {i: ~np.isnan(v) for i, v in enumerate(vecs)} # id of each track: vectors
            vecs_ret = {i: v[nan_dict[i][:, 0]] for i, v in enumerate(vecs)}
            points_ret = {i: p[nan_dict[i][:, 0]] for i, p in enumerate(points)}
            frames_ret = {i: np.arange(start_frame, end_frame-window_size, dtype=int)[nan_dict[i][:, 0]] for i,v in enumerate(vecs)}
            # cleaning up empty ids:
            remove_empty_entries([vecs_ret, points_ret, frames_ret], dtype="list")
        else:
            vecs = np.reshape(vecs, (n_vecs, tracks.shape[2]))
            points = np.reshape(points, (n_vecs, tracks.shape[2]))
            frames = np.tile(np.arange(start_frame, end_frame - window_size, dtype=int), tracks.shape[0])
            # filtering all frames without vectors
            nan_mask = ~np.isnan(vecs)
            vecs_ret = vecs[nan_mask[:, 0]]
            points_ret = points[nan_mask[:, 0]]
            frames_ret = frames[nan_mask[:, 0]]
    return vecs_ret, points_ret, frames_ret

def get_orientation_line(db):

    with OpenDB(db) as db_l:
        e_points = [(m.x, m.y) for m in db_l.getMarkers(type="elips")]
        straight_line = np.array(e_points[0])-np.array(e_points[3])

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