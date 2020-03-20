## plotting a movement vectors and angles in a series of frames
## showing that angles are calculated correctly
## this takes a while
import sys
import clickpoints
sys.path.append("/home/user/Software/fiber-orientation")
from database_functions import *
from plotting import *

folder="/home/user/Software/fiber-orientation/spheroid_spheroid_axis"
db = clickpoints.DataFile(os.path.join(folder,"db.cdb"),"r")
e_points=[(m.x,m.y) for m in db.getMarkers(type="elips")]
im=db.getImage().data
shape=im.shape
major_axis=np.linalg.norm(np.array(e_points[0])-np.array(e_points[3]))/2
minor_axis=np.linalg.norm(np.array(e_points[1])-np.array(e_points[2]))/2
center=np.mean(np.array(e_points),axis=0)
straight_line=np.array(e_points[0])-np.array(e_points[3])

#
start_frame = 0
end_frame = 30
plot_angles_database(db, start_frame, end_frame, straight_line, folder = "/home/user/Software/fiber-orientation/spheroid_spheroid_axis/angle_plot_tracks")
db.db.close()