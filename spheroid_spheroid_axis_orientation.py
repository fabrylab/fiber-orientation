# migration angles towards spheroid-spheroid angles
# produces images and videos of spatial angle distribution
# and histogram of angles in a selected area

#from pyTFM import utilities_TFM
import sys
import re
sys.path.append("/home/user/Software/fiber-orientation")
from plotting import plot_binned_angle_fileds
from database_functions import *

folder="/home/user/Software/fiber-orientation/spheroid_spheroid_axis"
db = clickpoints.DataFile(os.path.join(folder,"db.cdb"),"r")
e_points=[(m.x,m.y) for m in db.getMarkers(type="elips")]
im=db.getImage().data
shape=im.shape
major_axis=np.linalg.norm(np.array(e_points[0])-np.array(e_points[3]))/2
minor_axis=np.linalg.norm(np.array(e_points[1])-np.array(e_points[2]))/2
center=np.mean(np.array(e_points),axis=0)
straight_line=np.array(e_points[0])-np.array(e_points[3])

remove_single_img = True
max_frame=db.getImageCount() # that would be all frames

vecs, points, frames =  read_tracks_list_by_frame(db, end_frame=max_frame)
db.db.close()

hist_mask = np.load(os.path.join(folder,"hist_mask.npy"))
new_folder = createFolder(os.path.join(folder,"out"))

plot_binned_angle_fileds(vecs, points, frames, straight_line , binsize_time=30, step_size=2, bins_space=20,
                     folder=new_folder,e_points=e_points,plot_line=True,hist_mask=hist_mask,im=im, max_frame=max_frame)

# make a video with ffmpeg
command= 'ffmpeg -s 1000x796 -framerate 10 -y -i "%s"  -vcodec mpeg4 -b 10000k  "%s"'%(os.path.join(new_folder,"frame%04d.png"),os.path.join(new_folder,"out.mp4"))
os.system(command)
command= 'ffmpeg -s 1000x796 -framerate 10 -y -i "%s"  -vcodec mpeg4 -b 10000k  "%s"'%(os.path.join(new_folder,"hist_frame%04d.png"),os.path.join(new_folder,"out_hist.mp4"))
os.system(command)

if remove_single_img:
    files=os.listdir(new_folder)
    frames_files=[x for x in files if re.search("^(?!_)frame\d{4}.png",x)]
    hist_files=[x for x in files if re.search("hist_frame\d{4}.png",x)]
    for f in hist_files+frames_files:
        os.remove(os.path.join(new_folder,f))


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