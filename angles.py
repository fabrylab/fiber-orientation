import numpy as np
import copy
def radial_angle_distribution(points,center,angles,bins):
    rs = np.linalg.norm(points - center, axis=1)
    h1, bins1 = np.histogram(rs, bins=bins, weights=angles)
    h2, bins2 = np.histogram(rs, bins=bins)
    hist = h1 / h2
    bins = (bins1[:-1] + bins1[1:]) / 2
    return bins, hist

def spatial_angle_distribution(points,angles,bins=None):
    if not isinstance(bins,(np.ndarray,int,float)):
        bins = 10 # defuatl value innp.histogram2d
    hist1, bins1x, bins1y = np.histogram2d(points[:,0],points[:,1],weights=angles)
    hist2, bins2x, bins2y = np.histogram2d(points[:, 0], points[:, 1])
    hist=np.swapaxes(hist1/hist2,axis1=0,axis2=1) # hist returns x,y array, im is in y,x array
    return hist

def project_angle(a):
    # projects angles in range [0,2pi] to [0,pi/2] to extract measure of parallelicity
    angle=copy.deepcopy(a)
    if isinstance(angle,(float,int)):
        angle = np.array([angle])
    if not isinstance(angle,np.ndarray):
        angle = np.array(angle)
     #################### please check previous angle analysis

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
