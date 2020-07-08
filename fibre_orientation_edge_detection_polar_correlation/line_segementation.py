
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import clickpoints
from skimage import feature
from PIL import Image
from scipy.ndimage.filters import uniform_filter, median_filter, gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize,remove_small_objects
import cv2
from skimage.measure import regionprops


def normalizing(img,lq=0,uq=100):
    img = img - np.percentile(img, lq)  # 1 Percentile
    img = img / np.percentile(img,uq)  # norm to 99 Percentile
    img[img < 0] = 0.0
    img[img > 1] = 1.0
    return img


im = np.asarray(Image.open("/home/user/Desktop/biophysDS/dboehringer/Platte_3/Twitching-Experiments/Confocal-Experiments/2020-02-12-LuB1-Timelapse/stack 1/pos002/Pos002_S001_t314_z6_ch00.tif").convert("L"))
db = clickpoints.DataFile("/home/user/Desktop/mask_spheroid_david.cdb")
mask = db.getMask(frame=0).data.astype(bool)
db.db.close()
#im = ndi.gaussian_filter(im, 4)
# Compute the Canny filter for two values of sigma
im = normalizing(im, lq=10, uq=90)
plt.figure()
plt.imshow(im)
med_filter = median_filter(im, size = 30)
im_f = im - med_filter
plt.figure();plt.imshow(im_f)



#edges = feature.canny(im, sigma=i,mask=~mask.astype(bool))
#plt.figure()
#plt.imshow(im)
edges = im_f > threshold_otsu(im_f[~mask.astype(bool)])
edges[mask]=False
edges=skeletonize(edges)
edges=remove_small_objects(edges,2)
edges_show=np.ones(edges.shape)
edges_show[~edges]=np.nan
plt.imshow(edges_show,vmin=0,vmax=1,cmap="bwr")

plt.imshow(edges_show[200:400,200:400],vmin=0,vmax=1,cmap="bwr")
#lines = cv2.HoughLinesP(edges[200:400,200:400].astype("uint8"),0.1,np.pi/180,threshold=5,minLineLength=8,maxLineGap=2)
lines = cv2.HoughLines(edges[200:400,200:400].astype("uint8"),0.1,np.pi/180,threshold=20)
#for l in lines:
#    x1, y1, x2, y2 = l[0]
#    plt.plot([x1,x2],[y1,y2])

#plt.ylim((0,200))

#plt.xlim((0,200))

for l in lines:
    rho, theta=l[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 100 * (-b))
    y1 = int(y0 + 100 * (a))
    x2 = int(x0 - 100 * (-b))
    y2 = int(y0 - 100 * (a))
    plt.plot([x1, x2], [y1, y2])
plt.ylim((0,200))

plt.xlim((0,200))
###########
