import numpy as np
import matplotlib.pyplot as plt
import os
from fibre_orientation import normalizing
from itertools import product
from tqdm import tqdm
from pyTFM.plotting import show_quiver
## "second momemt tensor"
# following  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2691583/
#M=[sum((x-xmena)**2), sum((x-xmean)*(y-ymena))]
 #  [sum((x-xmean)*(y-ymena)), sum((y-ymena)**2)]


def sec_moment_tensor(arr):
    cent = np.array([arr.shape[1], arr.shape[0]]) / 2  # in x,y order
    y_pos, x_pos = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]),indexing="ij")
    x_pos, y_pos = x_pos.astype(float), y_pos.astype(float)
    x_pos -= cent[0]
    y_pos -= cent[1]
    # m1 = arr*(x_pos**2)
    # m2 = arr*x_pos*y_pos
    # m3 = arr*(y_pos**2)
    M = np.array([[np.sum(arr * (x_pos ** 2)), np.sum(arr * x_pos * y_pos)],
                  [np.sum(arr * x_pos * y_pos), np.sum(arr * (y_pos ** 2))]])
    return M


def extract_orientation(moment_tensor):
    values, vectors = np.linalg.eig(moment_tensor)
    return values[0], values[1], vectors[:, 0], vectors[:, 1]


def plot_orientation(arr):
    M = sec_moment_tensor(arr)
    val1, val2, vec1, vec2 = extract_orientation(M)
    max_v = np.maximum(val1, val2)
    val1 = val1 / max_v
    val2 = val2 / max_v
    plt.figure()
    plt.imshow(arr)
    center = np.array([arr.shape[1], arr.shape[0]]) / 2
    lf = np.mean(arr.shape)*0.3
    w = np.mean(arr.shape)*0.01
    plt.arrow(center[1], center[0], vec1[0] * val1 * lf, vec1[1] * val1 * lf, width=w)  # first eigenvector
    plt.arrow(center[1], center[0], vec2[0] * val2 * lf, vec2[1] * val2 * lf, width=w)  # second eigenvector
    plt.text(0,0,"val1=%s,val2=%s"%(np.round(val1,2),np.round(val2,2)))
    plt.show()


def orientation_field(arr, ws, ss):
    xs = np.arange(arr.shape[1])[np.arange(arr.shape[1]) % ss == 0]
    ys = np.arange(arr.shape[0])[np.arange(arr.shape[0]) % ss == 0]
    pos_x_dict = {x:i for i,x in enumerate(xs)}
    pos_y_dict = {y: i for i, y in enumerate(ys)}
    c_pos = list(product(xs, ys))
    ws_range = np.arange(ws).astype(int)

    max_vec_x = np.zeros((ys.shape[0]+1, xs.shape[0]+1))
    max_vec_y = np.zeros((ys.shape[0]+1, xs.shape[0]+1))
    ratio = np.zeros((ys.shape[0]+1, xs.shape[0]+1))

    for xp, yp in tqdm(c_pos):
        coord_y, coord_x = np.meshgrid(ws_range + yp, ws_range + xp, indexing="ij")
        try:
            M = sec_moment_tensor(arr[coord_y, coord_x])
            val1, val2, vec1, vec2 = extract_orientation(M)
            max_v = np.maximum(val1, val2)
            min_v = np.minimum(val1, val2)
            vec_max = vec1 if val1 > val2 else vec2
            r = max_v / min_v
            r = 100 if r>100 else r # pretty slopy
            ratio[pos_y_dict[yp], pos_x_dict[xp]] = r
            max_vec_x[pos_y_dict[yp], pos_x_dict[xp]] = vec_max[0] * r
            max_vec_y[pos_y_dict[yp], pos_x_dict[xp]] = vec_max[1] * r
        except IndexError:
            pass

    return max_vec_x, max_vec_y, ratio






arr = np.zeros((100, 100))
arr[list(range(0,100)),list(range(0,100))] = 1

ws = 6
ss = 1

max_vec_x, max_vec_y, ratio = orientation_field(arr, ws, ss)
#f = np.logical_or(np.isnan(ratio), ratio < 2)
#max_vec_x[f] = np.nan
#max_vec_y[f] = np.nan
show_quiver(max_vec_x, max_vec_y)
plt.figure()
plt.imshow(arr)


image = "/home/user/Desktop/biophysDS/dboehringer/Platte_3/Migration-and-fiberorientation/Evaluation-Andi-David/Fiber orientation/2 - polar trafo correlation/testing_orientation/Pos002_S001_t314_z6_ch00.tif"
image = "/home/user/Desktop/biophysDS/dboehringer/Platte_3/Migration-and-fiberorientation/Evaluation-Andi-David/Fiber orientation/2 - polar trafo correlation/testing_orientation/Series001_z000_ch00.tif"
image = "/home/user/Desktop/fibre_orientation_test/5.jpg"
im = plt.imread(image)
arr = normalizing(np.mean(im,axis=2))

#### should work well as substitute for single fibre detection
##### but has some issue still--> maybe programming error
##### more like ely needs some factor/weighting to avoid window wererelevant pixels are at the edges
### -> e.g _weight ifactor A*np.sqrt(x,y)/np.sum(A)

ws = 10
ss = 3

max_vec_x, max_vec_y, ratio = orientation_field(arr, ws, ss)
#f = np.logical_or(np.isnan(ratio), ratio < 1)
#max_vec_x[f] = np.nan
#max_vec_y[f] = np.nan
show_quiver(max_vec_x, max_vec_y,filter=[1.4,1])
plt.figure()
plt.imshow(arr)





folder="/home/user/Desktop/fibre_orientation_test/"
files=os.listdir(folder)
for f in files:
    im = plt.imread(os.path.join(folder,f))
    im = np.mean(im,axis=2)
    im = normalizing(im)
    im = np.abs(im-1)

arr = np.zeros((100, 100))
arr[:,50] = 1

ws = 6
ss = 1

max_vec_x, max_vec_y, ratio = orientation_field(arr, ws, ss)
#f = np.logical_or(np.isnan(ratio), ratio < 2)
#max_vec_x[f] = np.nan
#max_vec_y[f] = np.nan
show_quiver(max_vec_x, max_vec_y)
plt.figure()
plt.imshow(arr)



#### conclusion--> relies the "fibre" beeing orientated inthe middle of the window --> might not be so good??
## illustrate here:






for s in range(0,100,10):
    arr = np.zeros((100, 100))
    arr[s]=1
    plot_orientation(arr)


## in the paper: only keep "high cointrast" regions (max(val1,val2)/min(val1,val2) >2)


