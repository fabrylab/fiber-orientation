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
from matplotlib import patches


def normalizing(img,lq=0,uq=100):
    img = img - np.percentile(img, lq)  # 1 Percentile
    img = img / np.percentile(img,uq)  # norm to 99 Percentile
    img[img < 0] = 0.0
    img[img > 1] = 1.0
    return img



# coordinates
def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def polar_coordinate_transform(im, center, radius_res=2000, angle_res=2000):
    '''

    :param center: center of the spheroid in [x,y] or "iamge
    :param radius_res: number of points (resolution) used for the radii
    :param angle_res:  number of points (resolution) used for the angles
    :return:
    '''

    if isinstance(center,str):
        if center=="image":
            center = np.array([im.shape[1]/2,im.shape[0]/2])

    # generating a grid of angles and radii

    # pixel_size_input = pixel_size_out  * r_factor
    r_factor = ((np.sqrt(2)*np.max(im.shape)/2)*1.2)/radius_resactor

    rs = np.linspace(0,(np.sqrt(2)*np.max(im.shape)/2)*1.2,radius_res)
    phis = np.linspace(0,2*np.pi,angle_res)
    r_grid, phi_grid = np.meshgrid(rs, phis)

    # finding the cartesian coordinates corresponding to the grid
    xv, yv = pol2cart(r_grid, phi_grid)
    xv = np.round(xv + center[1]).astype(int)
    yv = np.round(yv + center[0]).astype(int)

    # grid for array of polar coordinates
    r_g, ph_g=np.meshgrid(np.arange(0,radius_res,dtype=int), np.arange(0,angle_res,dtype=int))

    # filter points that would lie outside of the image
    p_out_mask=(xv>=0) * (xv<im.shape[1]) * (yv>=0) * (yv<im.shape[0])
    xv = xv[p_out_mask]
    yv = yv[p_out_mask]
    r_g = r_g[p_out_mask]
    ph_g = ph_g[p_out_mask]

    # filling an array of polar coordinates
    polar_array=np.zeros((radius_res,angle_res))+np.nan
    polar_array[r_g, ph_g] = im[yv,xv]

    # finding a maximum radius where points for all angles are defined
    max_radius = np.min(np.where(np.isnan(polar_array))[0])

    return polar_array, max_radius, center, r_factor

def display_slices(start,window_size, window_dist, polar_array):
    r1 = [start, window_size + start]
    r2 = [start + window_dist, window_size + start + window_dist]
    #slice1 = polar_array[r1[0]:r1[1], :]
    #slice2 = polar_array[r2[0]:r2[1], :]

    fig=plt.figure();plt.imshow(polar_array)
    pa1=patches.Rectangle(xy=[0,r1[1]],width=2000,height=window_size,fill=False,edgecolor="red",linewidth=2)
    plt.gca().add_patch(pa1)
    pa2=patches.Rectangle(xy=[0,r2[1]],width=2000,height=window_size,fill=False,edgecolor="yellow",linewidth=2)
    plt.gca().add_patch(pa2)
    return fig


if __name__ == '__main__':
        
        
    # fibres with spheroid
    im = np.asarray(Image.open(r"\\131.188.117.96\biophysDS\dboehringer\Platte_3\Twitching-Experiments\Confocal-Experiments\2020-02-12-LuB1-Timelapse\stack 1\\pos002\\Pos002_S001_t314_z6_ch00.tif").convert("L"))
    db = clickpoints.DataFile(r"\\131.188.117.96\biophysDS\dboehringer\Platte_3\Migration-and-fiberorientation\Evaluation-Andi-David\Fiber orientation\2 - polar trafo correlation\testing_orientation/mask_spheroid_david.cdb")
    mask = db.getMask(frame=0).data.astype(bool)
    db.db.close()
    center=regionprops(mask.astype(int))[0].centroid
    
    
    polar_array, max_radius, center = polar_coordinate_transform(im, center, radius_res=2000, angle_res=2000)
    #ax_factor = r_factor * pixel_size # y_axis[pixel]*ax_factor --> y_axis[µn]
    
    
    
    #plt.figure();plt.imshow(im)
    #plt.figure();plt.imshow(polar_array)
    
    #correlation coefficient
    window_size = 30
    window_dist = 30
    cors=[]
    for i in range(2000-window_size-window_dist):
        r1 = [i,window_size+i]
        r2 = [i+window_dist,window_size+i+window_dist]
        slice1 = polar_array[r1[0]:r1[1],:]
        slice2 = polar_array[r2[0]:r2[1],:]
    
        nan_slice=np.logical_or(np.isnan(slice2),np.isnan(slice1))
        n=np.sum(~nan_slice)
        if n>1000:
            cor=np.corrcoef(slice1[~nan_slice],slice2[~nan_slice])[0][1]
        else:
            cor=np.nan
        cors.append(cor)
    
    
    #fig,axs=plt.subplots(1,2, gridspec_kw={'width_ratios': [3, 1]})
    fig3 = plt.figure(figsize=(9,6),constrained_layout=True)
    gs = fig3.add_gridspec(1, 10)
    ax1 = fig3.add_subplot(gs[0, :7])
    ax2 = fig3.add_subplot(gs[0, 7:])
    ax2.set_yticks([])
    axs=[ax1,ax2]
    axs[0].imshow(polar_array)
    axs[1].plot(cors,list(range(2000-window_size-window_dist)),color="C0")
    axs[1].set_ylim(2000,0)
    axs[1].set_xlim(-0.2,1)
    axs[0].set_ylim(2000,0)
    #plt.savefig("/home/user/Desktop/test.png")
    
    
    
    # reference curve
    im_ref = np.asarray(Image.open(r"\\131.188.117.96\biophysDS\dboehringer\Platte_3\Migration-and-fiberorientation\2019-07-19_Collagen_Poresize\neu1.2\1/Series001_z000_ch00.tif").convert("L"))
    polar_array, max_radius, center= polar_coordinate_transform(im_ref, center="image",radius_res=2000, angle_res=2000)
    #plt.figure();plt.imshow(im)
    #plt.figure();plt.imshow(polar_array)
    
    
    window_size = 30
    window_dist = 30
    cors_ref=[]
    for i in range(2000-window_size-window_dist):
        r1 = [i,window_size+i]
        r2 = [i+window_dist,window_size+i+window_dist]
        slice1 = polar_array[r1[0]:r1[1],:]
        slice2 = polar_array[r2[0]:r2[1],:]
    
        nan_slice=np.logical_or(np.isnan(slice2),np.isnan(slice1))
        n=np.sum(~nan_slice)
        if n>2000:
            cor=np.corrcoef(slice1[~nan_slice],slice2[~nan_slice])[0][1]
        else:
            cor=np.nan
        cors_ref.append(cor)
    
    axs[1].plot(cors_ref,list(range(2000-window_size-window_dist)),color="C1")
    mean_cor1=np.round(np.nanmean(cors[500:750]),2) # near spheroid
    mean_cor2=np.round(np.nanmean(cors[800:1400]),2) # middle distanace
    mean_cor3=np.round(np.nanmean(cors[500:]),2) # everything outside of sperhoid
    mean_cor_ref=np.round(np.nanmean(cors_ref),2)
    axs[1].text(0.5,150,"R near = "+str(mean_cor1),color="C0", fontsize=9)
    axs[1].text(0.5,200,"R far = "+str(mean_cor2),color="C0", fontsize=9)
    axs[1].text(0.5,250,"R all = "+str(mean_cor3),color="C0", fontsize=9)
    axs[1].text(0.5,100,"R ref = "+str(mean_cor_ref),color="C1", fontsize=9)
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.tick_params(direction="in", pad = -15)
    ax2.set_xlabel('Pearson Correlation Coefficient', fontsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    #plt.savefig('test.png')
    
    
    """
    Loop thorugh imager series
    """
    from glob import glob
    
    
    
    
    im_list = glob(r"\\131.188.117.96\biophysDS\dboehringer\Platte_3\Twitching-Experiments\Confocal-Experiments\2020-02-12-LuB1-Timelapse\stack 1\\pos002\\Pos002_S001_t*_z6_ch00.tif")
    
    
    
    
    
    
    for c,m in enumerate(im_list):
        
        fig4 = plt.figure(figsize=(9,6),constrained_layout=True)
        gs = fig4.add_gridspec(1, 10)
        ax1 = fig4.add_subplot(gs[0, :7])
        ax2 = fig4.add_subplot(gs[0, 7:])
        center=regionprops(mask.astype(int))[0].centroid
        
        im = normalizing(np.asarray(Image.open(m).convert("L") ),1,99)
        
        # plt.imshow(im)
        # plt.clf()
        
        polar_array, max_radius, center = polar_coordinate_transform(im, center, radius_res=2000, angle_res=2000)
    
        
        cors=[]
        for i in range(2000-window_size-window_dist):
            r1 = [i,window_size+i]
            r2 = [i+window_dist,window_size+i+window_dist]
            slice1 = polar_array[r1[0]:r1[1],:]
            slice2 = polar_array[r2[0]:r2[1],:]
        
            nan_slice=np.logical_or(np.isnan(slice2),np.isnan(slice1))
            n=np.sum(~nan_slice)
            if n>1000:
                cor=np.corrcoef(slice1[~nan_slice],slice2[~nan_slice])[0][1]
            else:
                cor=np.nan
            cors.append(cor)
    
    
        
        ax2.set_yticks([])
        axs=[ax1,ax2]
        axs[0].imshow(polar_array)  #, cmap='jet'
        axs[1].plot(cors,list(range(2000-window_size-window_dist)),color="C0")
        axs[1].set_ylim(2000,0)
        axs[1].set_xlim(-0.2,1)
        axs[0].set_ylim(2000,0)
            
      
        
        
        
        axs[1].plot(cors_ref,list(range(2000-window_size-window_dist)),color="C1")
        mean_cor1=np.round(np.nanmean(cors[500:750]),2) # near spheroid
        mean_cor2=np.round(np.nanmean(cors[800:1400]),2) # middle distanace
        mean_cor3=np.round(np.nanmean(cors[500:]),2) # everything outside of sperhoid
        mean_cor_ref=np.round(np.nanmean(cors_ref),2)
        axs[1].text(0.5,150,"R near = "+str(mean_cor1),color="C0", fontsize=7)
        axs[1].text(0.5,200,"R far = "+str(mean_cor2),color="C0", fontsize=7)
        axs[1].text(0.5,250,"R all = "+str(mean_cor3),color="C0", fontsize=7)
        axs[1].text(0.5,100,"R ref = "+str(mean_cor_ref),color="C1", fontsize=7)
        
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.tick_params(direction="in", pad = -15)
        ax2.set_xlabel('Pearson Correlation Coefficient', fontsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
    
    
        plt.savefig('series-test//'+str(c).zfill(4)+'.png')
    
        
    
        plt.clf()



