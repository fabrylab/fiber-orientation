import numpy as np
import matplotlib.pyplot as plt
import os
from pyTFM.plotting import show_quiver
from pyTFM.stress_functions import contractillity
from tqdm import tqdm
import re
import copy
from skimage.filters import gaussian
folder = "/home/user/Desktop/biophysDS/dboehringer/Platte_3/Evaluate_Angle_Contractility/LuB25_spheroids/Lub25_S2_190403/pos08_win50/"


drange=list(range(1,130))
conts=[]
for x in tqdm(drange):

    a = np.load(os.path.join(folder, "dis%s.npy"%str(x).zfill(6)), allow_pickle=True)
    a = a.item()
    u, v = a["u"], -a["v"] ### minus signe because of PIV, might change for later versions
    if x % 5 == 0:
        show_quiver(u, v, filter=[0, 3])

    cont, *rest = contractillity(u, v, 1, ~np.logical_or(np.isnan(u), np.isnan(v)))
    conts.append(cont)
plt.figure()
plt.plot(conts)
plt.hlines(0,0,130)

use_range=[0,70]



def stack_deformation_from_folder(folder, use_range="all"):
    all_files = [x for x in os.listdir(folder) if re.search("dis(\d{1,8})\.npy", x)]
    numbers = np.array([re.search("dis(\d{1,8}).npy", x).group(1) for x in all_files])
    fs = [f for n,f in sorted(zip(numbers, all_files))]
    ns = [int(x) for x in sorted(numbers)]
    if not use_range=="all":
        ns=[x for x in ns if x >= use_range[0] and x < use_range[1]]

    # loading first array just to get the shape
    a = np.load(os.path.join(folder, "dis%s.npy" % str(ns[0]).zfill(6)), allow_pickle=True)
    a = a.item()
    u, v = a["u"], -a["v"]  ### minus signe because of PIV, might change for later versions
    u_tot = np.zeros(u.shape)
    v_tot = np.zeros(v.shape)

    for x in tqdm(ns):
        a = np.load(os.path.join(folder, "dis%s.npy" % str(x).zfill(6)), allow_pickle=True)
        a = a.item()
        u, v = a["u"], -a["v"]  ### minus signe because of PIV, might change for later versions
        u_tot += u
        v_tot += v
    return u_tot, v_tot
u_tot, v_tot = stack_deformation_from_folder(folder, use_range=[0,70])
u_tot, v_tot = gaussian(u_tot,sigma=3), gaussian(v_tot,sigma=3)
#show_quiver(u_tot,v_tot,scale_ratio=0.1,filter=[0,4]) ### consider burring u tot and v tot e.g. with gaussian

# this gradient should produce symmetric strain tensor if rotation is zero
du_y, du_x = np.gradient(u_tot)
dv_y, dv_x = np.gradient(v_tot)

gamma = (dv_y+du_y)/2

##### this part is so far just a guess ##### 
# maximum strain and direction
prince_strain_max = (du_x+dv_y)/2 + np.sqrt(((du_x+dv_y)/2 )**2 + (gamma/2)**2) #### this is problematic
prince_strain_min = (du_x+dv_y)/2 - np.sqrt(((du_x+dv_y)/2 )**2 + (gamma/2)**2) #### this is problematic
prince_strain = copy.deepcopy(prince_strain_max)
choose_min = np.abs(prince_strain_max)<np.abs(prince_strain_min)
prince_strain[choose_min] = prince_strain_min[choose_min]


# i get two angles of principal strain ?? which one to choose
prince_strain_angle1 = np.arctan(2*gamma/(du_x-dv_y))
prince_strain_angle2 = np.arctan(2*gamma/(du_x-dv_y)) + np.pi/2
prince_strain_angle = copy.deepcopy(prince_strain_angle1)
prince_strain_angle[choose_min]=prince_strain_angle2[choose_min]



pvec_x = np.cos(prince_strain_angle) * prince_strain
pvec_y = np.sin(prince_strain_angle) * prince_strain
show_quiver(pvec_x,pvec_y,filter=[0,3])
##### ------- ##### 

# strain towards spheroid center( could be ignoring a lot of things.....)
center = np.array(u_tot.shape)/2
r_vecs = np.meshgrid(np.arange(u_tot.shape[0]),np.arange(u_tot.shape[1]),indexing="xy")
r_vecs = [r_vecs[1].T - center[1], r_vecs[0].T - center[0]]
dists = np.linalg.norm( np.stack(r_vecs,axis=2),axis=2)
r_vecs =[r_vecs[0]/dists, r_vecs[1]/dists]
show_quiver(r_vecs[0],r_vecs[1],filter=[0,4])

perpendicular_r_vecs=[r_vecs[1],-r_vecs[0]]
show_quiver(perpendicular_r_vecs[0],perpendicular_r_vecs[1],filter=[0,4]) # why does it look so weird???---> maybe some colormap issue

# strain vector when moving towards the center
strain_to_center_x = du_x * perpendicular_r_vecs[0] + gamma * perpendicular_r_vecs[1]
strain_to_center_y = gamma * perpendicular_r_vecs[0] + dv_y * perpendicular_r_vecs[1]

# component of this strain vector towards the center

strain_to_center = strain_to_center_x*r_vecs[0]+strain_to_center_y*r_vecs[1]
plt.figure();plt.imshow(strain_to_center);plt.colorbar()
# whats the unit I guess:
# unit of the deformations/pixel_length ofdef image
