import matplotlib
import matplotlib.pyplot as plt
import sys
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm
sys.path.append("/home/user/Software/fiber-orientation")
from angles import *
from database_functions import *



def plot_angles_database(db, start_frame, end_frame, orient_line, folder = "angle_plot_tracks"):
    # produces a video that shows the movement vector of the cell, a second vector and the angle
    # to this vector

    # reading frames
    vecs, points, frames = read_tracks_list_by_frame(db,start_frame,end_frame)
    # calculating angles
    angs = calculate_angle(vecs, orient_line, axis=1)
    angs = project_angle(angs)
    plt.ioff()
    createFolder(folder)
    with OpenDB(db) as db_l:
        for frame in tqdm(range(start_frame,end_frame-1)):
            # filtering for the correct frame// could be rather slow// better would be to constt
            l_frame_mask=frames==frame
            l_p = points[l_frame_mask]
            l_vec = vecs[l_frame_mask]
            l_ang = angs[l_frame_mask]
            ol_vecs = np.repeat(np.expand_dims(orient_line, axis=0), len(l_p), axis=0)
            try:
                im = db_l.getImage(frame=frame).data
            except FileNotFoundError:
                im = None
            fig=vizualize_angles(l_ang, l_p, l_vec, ol_vecs, image=im, arrows=True, normalize=True, size_factor=100,
                           text=False, sample_factor=1,cbar_max_angle = np.pi/2)
            fig.savefig(os.path.join(folder,"frame%s.png"%str(frame).zfill(4)))
            plt.close(fig)
    plt.ion()
    command = 'ffmpeg -s 1000x796 -framerate 10 -y -i "%s"  -vcodec mpeg4 -b 10000k  "%s"' % (
                                 os.path.join(folder, "frame%04d.png"), os.path.join(folder, "angles.mp4"))
    os.system(command)


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
