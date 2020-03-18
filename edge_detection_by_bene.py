import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.measure import regionprops,label
from tqdm import tqdm_notebook as tqdm
import scipy.optimize as opt
import clickpoints
import matplotlib
import copy


def project_angle(a):
    # projects angles in range [-pi,pi] to [0,pi/2] to extract measure of parallelicity
    angle=copy.deepcopy(a)
    if isinstance(angle,(float,int)):
        angle = np.array([angle])
    if not isinstance(angle,np.ndarray):
        angle = np.array(angle)
    a_adjusted = np.abs(angle).astype(float)  #################### please check previous angle analysis
    a_adjusted[a_adjusted > (np.pi / 2)] = np.pi  -  a_adjusted[a_adjusted > (np.pi / 2)]
    return a_adjusted


def calculate_angle(vec1, vec2, axis=0):
    v1 = np.expand_dims(vec1, axis=0) if len(np.array(vec1).shape) == 1 else np.array(vec1)
    v2 = np.expand_dims(vec2, axis=0) if len(np.array(vec2).shape) == 1 else np.array(vec2)

    # angles of vector in [0,2pi] to the x axis
    angles1 = np.arctan2(np.take(v1, 1, axis), np.take(v1, 0, axis)) + np.pi # xycoordinates are reversed
    angles2 = np.arctan2(np.take(v2, 1, axis), np.take(v2, 0, axis)) + np.pi # xycoordinates are reversed
    # angles of vectors towards each other in [0,2pi]
    angle = angles1-angles2
    return angle


# laden des Bilds
im = cv2.imread("/home/user/Desktop/biophysDS/abauer/testing_orientation/Pos002_S001_t314_z6_ch00.tif")
minVal = 50
maxVal = 200

edges = cv2.Canny(im,minVal,maxVal)

labels = label(edges)

regions = regionprops(labels)
angles_rad= np.array([])
sizes = np.array([])
for region in regions:
    angles_rad= np.append(angles_rad,region.orientation)
    sizes= np.append(sizes,region.area)


def linear_model(x,m,t):
    return(m*x + t)

area_threshold = 10 # auch 5 möglich
all_lines=[]

for region in tqdm(regions):  ### even better would be to weight the moments with origninal image???
    if region.area > area_threshold:
        p_opt,p_cov = opt.curve_fit(linear_model,region.coords[:,1],region.coords[:,0],p0= [0,0])
        plt.plot(region.coords[:,1],linear_model(region.coords[:,1],p_opt[0],p_opt[1]))
        all_lines.append(np.array([region.coords[:,1],linear_model(region.coords[:,1],p_opt[0],p_opt[1])]))
        #
plt.axis("equal")



im = np.asarray(Image.open("/home/user/Desktop/Pos002_S001_t314_z6_ch00.tif").convert("L"))
db = clickpoints.DataFile("/home/user/Desktop/mask_spheroid_david.cdb")
mask = db.getMask(frame=0).data.astype(bool)
db.db.close()
center=regionprops(mask.astype(int))[0].centroid


l_vecs = np.array([[l[0][-1]-l[0][0],l[1][-1]-l[1][0]] for l in all_lines])
l_points = np.array([[l[0][0],l[1][0]] for l in all_lines])
l_lenght = np.linalg.norm(l_vecs,axis=1)
c_vecs = np.array([[center[0]-p[0],center[0]-p[1]] for p in l_points])









ta=calculate_angle(l_vecs,c_vecs,axis=1)
ta_adjusted = project_angle(ta)

vizualize_angles(ta_adjusted, l_points, l_vecs,c_vecs,image=im )

