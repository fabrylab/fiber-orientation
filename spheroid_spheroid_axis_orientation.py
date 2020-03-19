import clickpoints
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage.filters import gaussian_filter,uniform_filter
import matplotlib
import os
from tqdm import tqdm
from skimage.morphology import label
from skimage.measure import regionprops
from scipy.ndimage import zoom
import cv2
from scipy.signal import convolve2d
import copy
from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime
#from pyTFM import utilities_TFM
import sys
sys.path.append("/home/user/Software/fiber-orientation")
from plotting import add_colorbar,display_spatial_angle_distribution,vizualize_angles
from angles import calculate_angle, project_angle
from pyTFM.utilities_TFM import createFolder


def read_tracks_to_list(db, min_id=0, max_id=np.inf):
    #db path or db object
    ndb = False
    if not isinstance(db, clickpoints.DataFile):
        ndb=True
        db = clickpoints.DataFile(db,"r")
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
    if ndb:
        db.close()
    return all_vecs, all_points, all_frames

def plot_binned_angle_fileds(vecs, points, frames, orient_line, method="mean bin", binsize_time=200,step_size=100,bins_space=30,
                         folder="",e_points=None,plot_line=True, hist_mask=None,im=None):
    plt.ioff()
    max_time=np.max(frames)
    for i,s in tqdm(enumerate(range(0,max_time-binsize_time,step_size))):

        time_mask=np.logical_and(frames>=s,  frames<(s+binsize_time))
        vecs_bin = vecs[time_mask]
        points_bin = points[time_mask]
        angs =  calculate_angle(vecs_bin, orient_line, axis=1)
        angs=project_angle(angs)

        # angle evaluation
        #ol_vecs= np.repeat(np.expand_dims(orient_line,axis=0),len(vecs_bin),axis=0)
        #vizualize_angles(angs, points_bin, vecs_bin,ol_vecs,image=im, arrows=True, normalize=True, size_factor=100,
        #                 text=False, sample_factor=5)
        ##### plot this with actual tracks
            # 2d hist of angles
        fig1=display_spatial_angle_distribution(points_bin, angs,bins=bins_space
                    ,fig_paras={"figsize":(10,7.96511),"frameon":False},
                        imshow_paras={"cmap":"Greens","vmin":0,"vmax":np.pi/2,"origin":"upper"})
        plt.axis("off")
        ax = plt.gca()



        #ax.set_position([0, 0, 1, 1])
        #if plot_line:
        #    plt.plot([e_points[0][0], e_points[3][0]], [e_points[0][1], e_points[3][1]])
        #angle_field, angles_adjusted, positions = get_smooth_angle_distribution(tracks_bin, method=method,window_size=window_size,sigma=sigma,nan_size=nan_size,fill_nans=fill_nans,fill_nan_smooth=fill_nan_smooth,raw=raw)
        #add_colorbar(vmin=0, vmax=np.pi/2,ax=ax, cmap="Greens",cbar_style="clickpoints",cbar_width="4%",cbar_height= "70%",
        #             cbar_borderpad= 6.5, cbar_tick_label_size = 30, cbar_str="angle towards\nspheroid-\nspeheroid axis",cbar_axes_fraction = 0.2)
        ax.text(0.3, 0.9, "time: " + str(datetime.timedelta(seconds=s*60*2)) + " to " + str(datetime.timedelta(seconds=(s+binsize)*60*2)), transform=ax.transAxes,
                 fontsize=20)

        fig1.savefig(os.path.join(folder,"frame%s.png"%(str(i).zfill(4))))

        # filtering only angles in a selected area
        hist_mask=hist_mask.astype(bool)
        angles_mask=[a for a,p in zip(angles_adjusted,positions) if hist_mask[np.round(p).astype(int)[0],np.round(p).astype(int)[1]]]
        fig2 = plt.figure(figsize=(10, 7.96511), frameon=False)
        plt.hist(angles_mask,density=True,color="C1",edgecolor='black')
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().set_xticks(np.linspace(0, np.pi / 2, 8))
        plt.gca().set_xticklabels([r"$0$"] + [r"$\frac{%s\pi}{%s}$" % s for s in
                                           [("", "16"), ("", "8"), ("3", "16"), ("", "4"), ("5", "16"), ("3", "8"),
                                            ("", "2")]], fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel("density",fontsize=20)
        plt.ylim(0,1)
        plt.xlabel("angle to spheroid-spheroid axis",fontsize=20)
        plt.text(0.2, 1, "time: " + str(datetime.timedelta(seconds=s * 60 * 2)) + " to " + str(
            datetime.timedelta(seconds=(s + binsize) * 60 * 2)), transform=plt.gca().transAxes,
                 fontsize=20)
        fig2.savefig(os.path.join(folder, "hist_frame%s.png" % (str(i).zfill(4))))

        plt.close(fig)
        plt.close(fig2)

    plt.ion()


folder="/home/user/Software/fiber-orientation/spheroid_spheroid_axis"
db=clickpoints.DataFile(os.path.join(folder,"db.cdb"),"r")
e_points=[(m.x,m.y) for m in db.getMarkers(type="elips")]
im=db.getImage().data
shape=im.shape
major_axis=np.linalg.norm(np.array(e_points[0])-np.array(e_points[3]))/2
minor_axis=np.linalg.norm(np.array(e_points[1])-np.array(e_points[2]))/2
center=np.mean(np.array(e_points),axis=0)
straight_line=np.array(e_points[0])-np.array(e_points[3])

vecs, points, frames =  read_tracks_to_list(db,max_id=400)
db.db.close()


hist_mask=np.load(os.path.join(folder,"hist_mask.npy"))
new_folder = createFolder(os.path.join(folder,"out"))

plot_binned_angle_fileds(vecs, points, frames, straight_line , binsize_time=30, step_size=2, bins_space=20,
                     folder=folder,e_points=e_points,plot_line=True,hist_mask=hist_mask,im=im)
# make a video with ffmpeg
command= 'ffmpeg -s 1000x796 -framerate 10 -y -i "%s"  -vcodec mpeg4 -b 10000k  "%s"'%(os.path.join(new_folder,"frame%04d.png"),os.path.join(new_folder,"out.mp4"))
command= 'ffmpeg -s 1000x796 -framerate 10 -y -i "%s"  -vcodec mpeg4 -b 10000k  "%s"'%(os.path.join(new_folder,"hist_frame%04d.png"),os.path.join(new_folder,"out_hist.mp4"))
os.system(command)
#import shutil
#files=[f for f in os.listdir(folder) if re.search(".*_(\d{4})\.png",f)]
#files=sorted(files,key=lambda x: re.search(".*_(\d{4})\.png",x).group(1))
#or f in files:
    #number=re.search(".*_(\d{4})\.png",f).group(1)
    #shutil.move(os.path.join(folder,f),os.path.join(folder,number+".png"))





'''
c = matplotlib.cm.get_cmap("Greens")((angles_adjusted - 0) / (np.pi/2 - 0))  # normalization and creating a color range
plt.figure()
plt.scatter(positions_x,positions_y,s=3,c=c)
norm = matplotlib.colors.Normalize(vmin=0, vmax=np.pi/2)#sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap("Greens"), norm=norm)
plt.colorbar(sm)
plt.imshow(im,cmap="Greys_r")
plt.xlim(0,shape[1])
plt.ylim(0,shape[0])
plt.plot([e_points[0][0],e_points[3][0]],[e_points[0][1],e_points[3][1]])

grid_x, grid_y = np.mgrid[-shape[0]:shape[0], -shape[1]:shape[1]]
'''