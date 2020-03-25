import matplotlib
import matplotlib.pyplot as plt
import sys
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm
sys.path.append("/home/user/Software/fiber-orientation")
from angles import *
from database_functions import *
import datetime


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

def plot_mean_angles(as1,as2=None,y_label="mean angles",title="",la1="angles near axis",la2="angles far from axis"):

def vizualize_angles(angles, points, vecs1, vecs2, arrows=True, image=None, normalize=True, size_factor=10, text=True, sample_factor=10,cbar_max_angle=None, angle_in_degree=True):
    fig=plt.figure()
    plt.plot(as1, color="C0", label=la1)
    if isinstance(as2, (np.ndarray, list)):
        plt.plot(as2, color="C1", label=la2)
    plt.hlines(np.pi / 4, 0, len(as1))
    plt.xlabel("time steps")
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    return fig

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
    if angle_in_degree:
        cbar.ax.set_yticklabels([str(np.round(360*i/(2*np.pi))) for i in cbar.ax.get_yticks()])
        cbar.ax.set_ylabel("angle [°]",fontsize=20)
    else:
        cbar.ax.set_ylabel("angle [rad]",fontsize=20)
    return fig

def display_spatial_angle_distribution(points,angles,bins=None,fig_paras={},imshow_paras={"cmap":"Greens"},bg="Greys"):
    hist = spatial_angle_distribution(points,angles,bins=bins)
    fig=plt.figure(**fig_paras)
    if isinstance(bg,(str,tuple,list)): # plotting a background( is there no better way??
        plt.imshow(np.zeros(hist.shape)+1,vmin=0,vmax=2,cmap=bg)
    plt.imshow(hist,**imshow_paras)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
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

def plot_binned_angle_fileds(vecs, points, frames, orient_line, binsize_time=200, step_size=100, bins_space=30,
                         folder="", e_points=None, plot_line=True, hist_mask=None, im=None, max_frame=None):
    plt.ioff()
    max_frame = np.max(frames) if not isinstance(max_frame,(int,float)) else max_frame
    angs = calculate_angle(vecs, orient_line, axis=1)
    angs = project_angle(angs)
    hist_mask = hist_mask.astype(bool)

    for i,s in tqdm(enumerate(range(0,max_frame-binsize_time,step_size))):
        time_mask=np.logical_and(frames>=s,  frames<(s+binsize_time))
        vecs_bin = vecs[time_mask]
        points_bin = points[time_mask]
        angs_bin = angs[time_mask]

        ### angle evaluation --> see angle_illustration_spheroid_spheroid_axis.py
        t1 = str(datetime.timedelta(seconds=s * 60 * 2))  # 2 is frame rate in minutes
        t2 = str(datetime.timedelta(seconds=(s + binsize_time) * 60 * 2))
        fig1=spatial_sph_axis(points_bin, angs_bin, bins_space, t1, t2)
        fig1.savefig(os.path.join(folder,"frame%s.png"%(str(i).zfill(4))))
        plt.close(fig1)

        # filtering only angles in a selected area
        inside_slect = hist_mask[np.round(points_bin[:, 1]).astype(int), np.round(points_bin[:, 0]).astype(int)]
        angles_mask = angs_bin[inside_slect]
        fig2 = hist_for_angles(angles_mask, t1, t2)
        fig2.savefig(os.path.join(folder, "hist_frame%s.png" % (str(i).zfill(4))))
        plt.close(fig2)

    plt.ion()

def spatial_sph_axis(points,angs,bins_space,t1,t2):
    fig = display_spatial_angle_distribution(points, angs, bins=bins_space
                                              , fig_paras={"figsize": (10, 7.96511), "frameon": False,
                                                           "facecolor": "grey"},
                                              imshow_paras={"cmap": "Greens", "vmin": 0, "vmax": np.pi / 2,
                                                            "origin": "upper"})
    plt.axis("off")
    ax = plt.gca()
    # if plot_line:
    #    plt.plot([e_points[0][0], e_points[3][0]], [e_points[0][1], e_points[3][1]])
    ax.text(0.3, 0.9, "time: %s to %s" % (t1, t2), transform=ax.transAxes, fontsize=20)
    return fig

def hist_for_angles(angs,t1,t2):

    fig = plt.figure(figsize=(10, 7.96511), frameon=False)
    plt.hist(angs, density=True, color="C1", edgecolor='black')
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().set_xticks(np.linspace(0, np.pi / 2, 8))
    plt.gca().set_xticklabels([r"$0$"] + [r"$\frac{%s\pi}{%s}$" % s for s in
                                          [("", "16"), ("", "8"), ("3", "16"), ("", "4"), ("5", "16"), ("3", "8"),
                                           ("", "2")]], fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("density", fontsize=20)
    plt.ylim(0, 1)
    plt.xlabel("angle to spheroid-spheroid axis", fontsize=20)
    plt.text(0.2, 1, "time: %s to %s" % (t1, t2), transform=plt.gca().transAxes, fontsize=20)
    return fig