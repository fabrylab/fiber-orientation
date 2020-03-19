import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("/home/user/Software/fiber-orientation")
from angles import *


def vizualize_angles(angles, points, vecs1, vecs2, arrows=True, image=None, normalize=True, size_factor=10, text=True, sample_factor=10,cbar_max_angle=None):
    fig=plt.figure()
    if isinstance(image, np.ndarray):
        plt.imshow(image,cmap="Greys")
    # normalization and creating a color range
    colors = matplotlib.cm.get_cmap("Greens")(angles/np.max(angles))
    for i,(p,v1,v2,c) in enumerate(zip(points,vecs1,vecs2,colors)):
        if i % sample_factor == 0:
            if arrows:
                if normalize:
                    v1 = v1 * size_factor / np.linalg.norm(v1)
                    v2 = v2 * size_factor / np.linalg.norm(v2)
                plt.arrow(p[0], p[1], v1[0], v1[1], head_width=5, color="white")
                plt.arrow(p[0], p[1], v2[0], v2[1], head_width=5, color="C3")
                if isinstance(image, np.ndarray):
                    plt.ylim((image.shape[0],0))
                    plt.xlim((0,image.shape[1]))
            plt.scatter(p[0],p[1],color=c)
            if text:
                plt.text(p[0], p[1], str(i) + "\n" + str(np.round(angles[i],2)))
    vmax = cbar_max_angle if isinstance(cbar_max_angle,(int,float)) else np.max(angles)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap("Greens"), norm=norm)
    cbar=plt.colorbar(sm)
    cbar.ax.set_ylabel("angle [rad]",fontsize=20)
    return fig

def display_spatial_angle_distribution(points,angles,bins=None,fig_paras={},imshow_paras={"cmap":"Greens"}):
    hist = spatial_angle_distribution(points,angles,bins=bins)
    fig=plt.figure(**fig_paras)
    plt.imshow(hist,**imshow_paras)
    cbar=plt.colorbar()
    cbar.ax.set_ylabel("angle [rad]",fontsize=20)
    return fig

def diplay_radial_angle_distribution(points, center, angles, bins, plt_labels=["",""]):
    '''

    :param points: # list of arrays
    :param center: # list of arrays
    :param angles: # list of arrays
    :param bins:
    :return:
    '''


    fig=plt.figure()
    for p,c,a,pl in zip(points,center,angles,plt_labels):
        bins, hist = radial_angle_distribution(p, c, a, bins)
        plt.plot(bins, hist,label=pl)
        plt.xlabel("distance")
        plt.ylabel("mean angle")
        plt.hlines(np.pi / 4, np.min(bins), np.max(bins) * 1.2)
        plt.legend(loc="upper right")
    return fig


def add_colorbar(vmin,vmax, cmap,ax,cbar_style,cbar_width,cbar_height,cbar_borderpad,cbar_tick_label_size,cbar_str,cbar_axes_fraction):
    # adding colorbars inside or outside of plots
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap(cmap), norm=norm)
    if cbar_style == "clickpoints": # colorbar inside of the plot
        cbaxes = inset_axes(ax, width=cbar_width, height=cbar_height, loc=5, borderpad=cbar_borderpad)
        cb0 = plt.colorbar(sm,cax=cbaxes)
        cbaxes.set_title(cbar_str, color="white",pad=20,fontsize=20)
        cbaxes.tick_params(colors="white",labelsize=cbar_tick_label_size)
        cb0.set_ticks(np.linspace(0,np.pi/2,8),)
        cbaxes.set_yticklabels([r"$0$"]+[r"$\frac{%s\pi}{%s}$"%s for s in [("","16"),("","8"),("3","16"),("","4"),("5","16"),("3","8"),("","2")]])
    else: # colorbar outide of the plot
        cb0=plt.colorbar(sm, aspect=20, shrink=0.8,fraction=cbar_axes_fraction) # just exploiting the axis generation by a plt.colorbar
        cb0.outline.set_visible(False)
        cb0.ax.tick_params(labelsize=cbar_tick_label_size)
        cb0.ax.set_title(cbar_str, color="black")
