# functions handling accessing and writing to clickpoints
import clickpoints
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict
import sys
sys.path.append("/home/user/Software/fiber-orientation")



def createFolder(directory):
    '''
    function to create directories if they dont already exist
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

class OpenDB:
    # context manager for database file. if db is a path a new file handle is opened
    # this handle is later closed (!). if db is already clickpoints.DataFile object,
    # the handle is not closed
    def __init__(self,db, method="r"):
        if isinstance(db, clickpoints.DataFile):
            self.file = db
            self.db_obj=True
        else:
            self.file = clickpoints.DataFile(db, method)
            self.db_obj = False

    def __enter__(self):
        return self.file

    def __exit__(self, type, value, traceback):
        if not self.db_obj:
            self.file.db.close()


def read_tracks_list_by_frame(db, start_frame=0, end_frame=None):

    with OpenDB(db) as db_l:
        if not isinstance(end_frame,int):
            end_frame = db_l.getImageCount()
        tracks = db_l.getTracksNanPadded(start_frame=start_frame, end_frame=end_frame)
         # calcualting vectors
        n_vecs = tracks.shape[0] * (tracks.shape[1] - 1)
        vecs = np.reshape(tracks[:, 1:, :] - tracks[:, :-1, :], (n_vecs, tracks.shape[2]))
        points = np.reshape(tracks[:, :-1, :], (n_vecs, tracks.shape[2]))
        frames = np.tile(np.arange(start_frame, end_frame-1, dtype=int), tracks.shape[0])
         # filtering all frames without vectors
        nan_mask = ~np.isnan(vecs)
    return vecs[nan_mask[:,0]], points[nan_mask[:,0]], frames[nan_mask[:,0]]

def make_frame_dict(frames,*args):
    dict_list=[defaultdict(list) for i in range(args)]
    for f,[ar] in zip(frames,*args):
        for i,a in ar:
            dict_list[i].append(ar)
    return dict_list


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