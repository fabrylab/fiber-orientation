from fibre_orientation import *
import os
from tqdm import tqdm
folder = "/home/user/Desktop/biophysDS/dboehringer/Platte_3/Twitching-Experiments/Confocal-Experiments/2020-02-12-LuB1-Timelapse/stack 1/pos002/"
# ref_image nicht benötigt


db = clickpoints.DataFile(r"/home/user/Software/fiber-orientation/evaluation_polar_coordinates/mask_spheroid_david.cdb")
mask = db.getMask(frame=0).data.astype(bool)
db.db.close()
center=regionprops(mask.astype(int))[0].centroid


im_list=[x for x in os.listdir(folder) if "z6" in x]
window_size = 30
window_dist = 30

mcor1 = []
mcor2 = []
mcor3 = []




for c, m in tqdm(enumerate(im_list)):
    if not c%10==0:
        continue



for c, m in tqdm(enumerate(im_list)):
    if not c%10==0:
        continue

    im = normalizing(np.asarray(Image.open(os.path.join(folder,m)).convert("L")), 1, 99)
    polar_array, max_radius, center, r_factor = polar_coordinate_transform(im, center, radius_res=2000, angle_res=2000)
    # plt.imshow(im)
    # plt.clf()

    cors = []
    for i in range(2000 - window_size - window_dist):
        r1 = [i, window_size + i]
        r2 = [i + window_dist, window_size + i + window_dist]
        slice1 = polar_array[r1[0]:r1[1], :]
        slice2 = polar_array[r2[0]:r2[1], :]

        nan_slice = np.logical_or(np.isnan(slice2), np.isnan(slice1))
        n = np.sum(~nan_slice)
        if n > 1000:
            cor = np.corrcoef(slice1[~nan_slice], slice2[~nan_slice])[0][1]
        else:
            cor = np.nan
        cors.append(cor)

    mcor1.append(np.round(np.nanmean(cors[500:750]), 2))  # near spheroid
    mcor2.append(np.round(np.nanmean(cors[800:1400]), 2))  # middle distanace
    mcor3.append(np.round(np.nanmean(cors[500:]), 2))  # everything outside of sperhoid
plt.figure()
#plt.plot(mcor1, label="near correlation")
#plt.plot(mcor2, label="far correlation")
plt.plot(mcor3, label= "both")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
#plt.legend()
plt.ylim(0,0.5)