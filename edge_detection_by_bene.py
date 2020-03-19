# applying the edge detection by benedict on an image of fibres
# around a spheroid and on a random fibres
# needs evaluation_polar_coordinates/Pos002_S001_t314_z6_ch00.tif and
# evaluation_polar_coordinates/mask_spheroid_david.cdb
# evaluation_polar_coordinates/Series001_z000_ch00.tif

# produces plots: illustrating some of the angles, showing the spatial angle distribution
# in a heat map, and the radial angle distribution (100 bins) in both random and spheroid fibres
from PIL import Image
from skimage.measure import regionprops,label
import clickpoints
import sys
sys.path.append("/home/user/Software/fiber-orientation")
from plotting import *
from angles import *
from fibre_orientation import benes_edge_detection_method
#import os
#os.chdir("/home/user/Software/fiber-orientation")

# spheroid
im = np.asarray(Image.open("evaluation_polar_coordinates/Pos002_S001_t314_z6_ch00.tif").convert("L"))
l_vecs, l_points, l_length =  benes_edge_detection_method(im, minVal=50, maxVal=200, area_threshold=10)
db = clickpoints.DataFile("evaluation_polar_coordinates/mask_spheroid_david.cdb")
mask = db.getMask(frame=0).data.astype(bool)
db.db.close()
center=np.array(regionprops(mask.astype(int))[0].centroid)

c_vecs = np.array([[center[0]-p[0],center[1]-p[1]] for p in l_points])
ta = calculate_angle(l_vecs,c_vecs, axis=1)
ta_adjusted = project_angle(ta)
fig_angles1=vizualize_angles(ta_adjusted, l_points, l_vecs,c_vecs ,image=im ,sample_factor=4, size_factor=30, text=False, arrows=True)
#diplay_radial_angle_distribution(l_points,center,ta_adjusted, 100)
fig_angles_dist1=display_spatial_angle_distribution(l_points,ta_adjusted, bins=None)
plt.show()


# random fibres
im = np.asarray(Image.open("evaluation_polar_coordinates/Series001_z000_ch00.tif").convert("L"))
l_vecs_rand, l_points_rand, l_length_rand =  benes_edge_detection_method(im, minVal=50, maxVal=200, area_threshold=10)
center_rand = np.array([im.shape[0]/2,im.shape[1]/2])

c_vecs_rand = np.array([[center_rand[0]-p[0],center_rand[1]-p[1]] for p in l_points_rand])
ta_rand = calculate_angle(l_vecs_rand,c_vecs_rand,axis=1)
ta_adjusted_rand = project_angle(ta_rand)
fig_angles2=vizualize_angles(ta_adjusted_rand, l_points_rand, l_vecs_rand,c_vecs_rand ,image=im, sample_factor=4, size_factor=30, text=False, arrows=True)
#diplay_radial_angle_distribution(l_points_rand, center_rand, ta_adjusted_rand, 100)
fig_angles_dist2=display_spatial_angle_distribution(l_points_rand, ta_adjusted_rand, bins=None)
plt.show()


fig_rad = diplay_radial_angle_distribution([l_points,l_points_rand],[center,center_rand],
                                 [ta_adjusted,ta_adjusted_rand],100,plt_labels=["spherod","random fibres"])
# saving
fig_angles1.savefig("evaluation_polar_coordinates/angles_spheroid.png")
fig_angles_dist1.savefig("evaluation_polar_coordinates/angle_dist_spheroid.png")
fig_angles2.savefig("evaluation_polar_coordinates/angles_random.png")
fig_angles_dist2.savefig("evaluation_polar_coordinates/angle_dist_spheroid.png")
fig_rad.savefig("evaluation_polar_coordinates/radial_angle_dist.png")