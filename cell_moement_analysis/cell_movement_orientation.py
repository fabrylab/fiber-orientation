# migration angles towards spheroid-spheroid angles
# produces images and videos of spatial angle distribution
# and histogram of angles in a selected area

#from pyTFM import utilities_TFM
import sys
import re
import matplotlib.pyplot as plt
sys.path.append("/home/user/Software/fiber-orientation")
from plotting import plot_binned_angle_fileds, vizualize_angles, plot_mean_angles, display_spatial_angle_distribution, plot_distance_distribution
from database_functions import *
from cell_moement_analysis.angel_calculations import *
from contextlib import suppress
from utilities import *


def angles_to_straight_line(straight_line, vecs, points, frames):
    vecs_ar, points_ar, frames_ar = np.vstack(list(vecs.values())), np.vstack(list(points.values())), np.hstack(
        list(frames.values()))
    angs = calculate_angle(vecs_ar, straight_line, axis=1)
    angs = project_angle(angs)
    return vecs_ar, points_ar, frames_ar, angs



def  angles_to_centers(centers, vecs, points, frames):
    # converting to one big array
    vecs_ar, points_ar, frames_ar = np.vstack(list(vecs.values())), np.vstack(list(points.values())), np.hstack(list(frames.values()))
    # associate to point to closest center
    distances = np.linalg.norm(points_ar[:,None] - centers[None,:], axis=2) #axis0->points,axis1->centers
    mins = np.argmin(distances,axis=1)
    n = range(len(centers))
    point_dict = {i:points_ar[mins==i] for i in n}
    vec_dict = {i:vecs_ar[mins==i] for i in n}
    frames_dict = {i:frames_ar[mins==i] for i in n}
    dist_vectors = {i: point_dict[i] - centers[i] for i in n}

    angs_dict = {i:project_angle(calculate_angle(vec_dict[i], dist_vectors[i], axis=1)) for i in range(len(centers))}

    return point_dict, vec_dict, frames_dict, angs_dict, dist_vectors



def read_tracks_to_binned_dict(db, binsize_time, step_size, max_frame=None):
    straight_line = get_orientation_line(db)
    vecs, points, frames, angs = angles_to_straight_line(db, straight_line, max_frame=max_frame)
    max_frame = np.max(frames) if not isinstance(max_frame, (int, float)) else max_frame
    # might be to memory intensive like this
    vecs_b, points_b, angs_b = binning_by_frame(frames, max_frame, binsize_time, step_size, vecs, points, angs)

    return vecs_b, points_b, angs_b


def get_frames_list(db, min_frame=0, max_frame=None, ws_mean=30, bs_mean=2):
    if max_frame is None:
        with OpenDB(db) as db_l:
            max_frame = db_l.getImageCount()  #
    frames = np.arange(min_frame, max_frame - ws_mean, bs_mean)
    return frames


def make_spheroid_spheroid_orientation_vids(db_path,hist_mask,out_path):
    straight_line = get_orientation_line(db_path)
    remove_single_img = True

    db = clickpoints.DataFile(db_path, "r")
    im = db.getImage().data

    max_frame = db.getImageCount()  # that would be all frames

    vecs, points, frames = read_tracks_list_by_frame(db, end_frame=max_frame, track_types="all")
    db.db.close()
    hist_mask = np.load(hist_mask)
    new_folder = createFolder(out_path)

    plot_binned_angle_fileds(vecs, points, frames, straight_line, binsize_time=30, step_size=2, bins_space=20,
                             folder=new_folder, e_points=None, plot_line=True, hist_mask=hist_mask, im=im,
                             max_frame=max_frame)

    # make a video with ffmpeg
    command = 'ffmpeg -s 1000x796 -framerate 10 -y -i "%s"  -vcodec mpeg4 -b 10000k  "%s"' % (
    os.path.join(new_folder, "frame%04d.png"), os.path.join(new_folder, "out.mp4"))
    os.system(command)
    command = 'ffmpeg -s 1000x796 -framerate 10 -y -i "%s"  -vcodec mpeg4 -b 10000k  "%s"' % (
    os.path.join(new_folder, "hist_frame%04d.png"), os.path.join(new_folder, "out_hist.mp4"))
    os.system(command)

    if remove_single_img:
        files = os.listdir(new_folder)
        frames_files = [x for x in files if re.search("^(?!_)frame\d{4}.png", x)]
        hist_files = [x for x in files if re.search("hist_frame\d{4}.png", x)]
        for f in hist_files + frames_files:
            os.remove(os.path.join(new_folder, f))



def angle_to_center_analysis(db_path, output_folder, output_file, min_frame=0, max_frame=None, ws_angles=1, ws_mean=30, bs_mean=2, mark_center=True, fl=[], wl=[]):
    '''
    Calculates the movement angle of a cell towards the closest spheroid center. The angle is projected to a range
    between 0° to 90°. A cell moving away and a cell moving towards the spheroid center both have an angle of 0 degrees.
    Your database needs tracks of moving cells and markers of type "center", that mark the center of spheroids.
    There can be as manny centers as you like.
    :param db_path: full path to the database
    :param output_folder: output folder, is created on demand
    :param max_frame:  maximal frame up to which tracks are analyzed. None if you want all tracks.
    :param ws_angles:  angle of cell movement is calculated using the i th and (i+ws_angles) th position (in frames)
    of the cell
    :param ws_mean: all angles in a window of ws_mean frames are pooled by calculating the mean angle
    :param bs_mean: the window for the mean angle is moved bs_mean frames at each iteration. (so if
    ws_mean==bs_mean there would be no overlap)
    :param weighting: "nw" (no weighting) or "lw" (linear weigthing, angles are weighted by the length of teh
    cell movement step)
    :return:
    '''

    #reading center positions, one image of the spheroids and identifying maximal frame if max_frame==None
    createFolder(output_folder)
    out_file = os.path.join(output_folder, output_file)

    im,im_shape = None,None
    with OpenDB(db_path) as db:
        centers = np.array([(m.x, m.y) for m in db.getMarkers(type="center")])
        try:
            im = db.getImages()[0].data
            im_shape = im.shape
        except FileNotFoundError:
            print("couldn't find image and image shape for database " + db_path)
            im_shape = (2192, 2752)
        max_frame = max_frame if isinstance(max_frame, int) else db.getImageCount()


    # reading angles, movement vectors, frames and points adn randomizing cell tracks
    vecs, points, frames = read_tracks_list_by_frame(db_path, window_size=ws_angles, start_frame=min_frame, end_frame=max_frame,
                                                     return_dict=True, track_types="all") # reading vetctors and points as arrays

    # splitting into dictionary according to the closest spheroid center
    point_dict, vec_dict, frames_dict, angs_dict, dist_vectors = angles_to_centers(centers, vecs, points, frames)
    lengths_dict = {key: np.linalg.norm(v, axis=1) for key,v in vec_dict.items()}   # calculating the length

    # randomizing orientation and splitting into dictionary according to the closest spheroid center
    vecs_rot, points_rot, frames_rot = randomize_tracks(vecs, points, frames, im_shape)
    point_dict_rand, vec_dict_rand, frames_dict_rand, angs_dict_rand, dist_vectors_rand = angles_to_centers(centers, vecs_rot, points_rot, frames_rot)
    lengths_dict_rand = {key: np.linalg.norm(v, axis=1) for key, v in vec_dict_rand.items()}  # calculating the length

    point, vec, frames, angs, dists = flatten_dict(point_dict, vec_dict, frames_dict, angs_dict, dist_vectors)
    point_rand, vec_rand, frames_rand, angs_rand, dists_rand = flatten_dict(point_dict_rand, vec_dict_rand, frames_dict_rand,
                                                                            angs_dict_rand, dist_vectors_rand)
    # vizualizing angles
    fig_angel_viz = vizualize_angles(angs, point, vec, dists, arrows=True, image=im, normalize=True, size_factor=50,
                     text=False, sample_factor=int(np.ceil(0.2*max_frame)))
    fig_angel_viz.savefig(os.path.join(output_folder, "anlges_vizualization.png"))

    fig_angel_viz_rand = vizualize_angles(angs_rand, point_rand, vec_rand, dists_rand, arrows=True, image=im, normalize=True, size_factor=50,
                                     text=False, sample_factor=int(np.ceil(0.2*max_frame)))
    fig_angel_viz_rand.savefig(os.path.join(output_folder, "anlges_vizualization_randomized.png"))


    # filtering and wieghting:
    FW=FilterAndWeighting(angs_dict, lengths_dict, point_dict, frames_dict)
    FW.apply_filter(fl)
    FW.apply_weighting(wl)
    angs_dict, lengths_dict, point_dict, frames_dict = FW.angles, FW.lengths, FW.points, FW.frames

    FW_rand=FilterAndWeighting(angs_dict_rand, lengths_dict_rand, point_dict_rand, frames_dict_rand)
    FW.apply_filter(fl)
    FW.apply_weighting(wl)
    angs_dict_rand, lengths_dict_rand, point_dict_rand, frames_dict_rand = FW_rand.angles, FW_rand.lengths, FW_rand.points, FW_rand.frames

    # calculating mean angles over time
    mas = []
    ma_rands = []
    for key in frames_dict.keys():
        mas.append(get_mean_anlge_over_time(frames_dict[key], min_frame, max_frame, ws_mean, bs_mean,
                                            angs_dict[key]))
        ma_rands.append(get_mean_anlge_over_time(frames_dict_rand[key], min_frame, max_frame, ws_mean,
                                                 bs_mean, angs_dict_rand[key]))

    # plotting mean angles over time
    labels=["spheroid " + str(i+1) for i in range(len(mas))] + ["randomly reorientated"] * len(ma_rands)
    frames_out = np.arange(min_frame, max_frame - ws_mean)[::bs_mean]

    fig = plot_mean_angles(mas+ma_rands, frames_out, vmin=0, vmax=np.pi/2, labels=labels)
    fig.savefig(os.path.join(output_folder,"mean_anlge_over_time.png"))

    # saving mean and randomized angles
    lvs = [frames_out]+ mas+ ma_rands
    headers = ["frames"] + ["angle to sph %s" % str(i) for i in range(len(mas))] + \
              ["angle to sph %s randomized" % str(i) for i in range(len(mas))]
    arr_save = np.round(np.array([np.array(x) for x in lvs if isinstance(x,(list,np.ndarray))]),5).T
    np.savetxt(out_file, np.array(arr_save), delimiter=",", fmt="%.3f", header=",".join(headers))


    # 2d maps of angles
    bins = (int(10*im_shape[1]*2/(im_shape[0]+im_shape[1])),int(10*im_shape[0]*2/(im_shape[0]+im_shape[1])))
    fig_spatial = display_spatial_angle_distribution(point, angs, bins=bins, fig_paras={}, imshow_paras={"cmap": "Greens"}, bg="Greys", vmin=29*2*np.pi/360,vmax=55*2*np.pi/360)
    if mark_center:
        for c in centers:
            fig_spatial.get_axes()[0].plot(c[0] * bins[1] / im_shape[0], c[1] * bins[0] / im_shape[1], "+", color="red")

    fig_spatial_random = display_spatial_angle_distribution(point_rand, angs_rand, bins=bins, fig_paras={}, imshow_paras={"cmap": "Greens"}, bg="Greys", vmin=29*2*np.pi/360,vmax=55*2*np.pi/360)
    if mark_center:
        for c in centers:
            fig_spatial_random.get_axes()[0].plot(c[0] * bins[1] / im_shape[0], c[1] * bins[0] / im_shape[1], "+",
                                                  color="red")

    fig_spatial.savefig(os.path.join(output_folder,"spatial_distribution.png"))
    fig_spatial_random.savefig(os.path.join(output_folder,"spatial_distribution_random.png"))

    #fig_spatial = display_spatial_angle_distribution(point, angs, bins=(5,5), fig_paras={}, imshow_paras={"cmap": "Greens"}, bg="Greys", diff_to_random_angle=True)
    #fig_spatial_random = display_spatial_angle_distribution(point_rand, angs_rand, bins=(10,10), fig_paras={}, imshow_paras={"cmap": "Greens"}, bg="Greys", diff_to_random_angle=True)
    #fig_spatial.savefig(os.path.join(output_folder,"diff_spatial_distribution.png"))
    #fig_spatial_random.savefig(os.path.join(output_folder,"diff_spatial_distribution_random.png"))
    print(len(angs))
    print(np.nanmean(angs_rand))
    print(np.nanmean(angs))
    plt.close("all")

def angle_to_line_analysis(db_path, hist_mask_path, output_folder, output_file="mean_anlges.txt", min_frame=0, max_frame=None, ws_angles=1, ws_mean=30, bs_mean=2,
                           weighting = "nw"):
    '''
    Calculates the movement angle of a cell with respect to a line for example connecting two spheroids. The line is
    obtained from two markers of type "straight_line" in the clickpoints data base. Only one line is supported
    per database. Angles are split in two areas defined by a mask (.npy file) in "hist_mask_path". The mask is not
    filled automatically. Angles inside and outside of this area are referred to as "in" and "out".
    Angles are binned over ws_mean frames (see below) and stored in a text file.
    :param db_path: path to clickpoints database containing tracks and "straight_line" markers
    :param hist_mask_path: path to .npy file containing a mask, marking a region of interest
    :param output_folder: output folder, is generated automatically, use absolute paths
    :param output_file: .txt output file
    :param max_frame: Maximal frame up to which tracks are analyzed. Use None for all frames.
    :param ws_angles:  angle of cell movement is calculated using the i th and (i+ws_angles) th position (in frames)
    of the cell
    :param ws_mean: all angles in a window of ws_mean frames are pooled by calculating the mean angle
    :param bs_mean: the window for the mean angle is moved bs_mean frames at each iteration. (so if
    ws_mean==bs_mean there would be no overlap)
    :param weighting: "nw" (no weighting) or "lw" (linear weigthing, angles are weighted by the length of teh
    cell movement step)
    :return:
    '''

    createFolder(output_folder)
    out_file = os.path.join(output_folder,output_file)
    with OpenDB(db_path) as db:
        with suppress(FileNotFoundError): im = db.getImages()[0].data
        e_points = [(m.x, m.y) for m in db.getMarkers(type="straight_line")]
        straight_line = np.array(e_points[0]) - np.array(e_points[1])
        max_frame = max_frame if isinstance(max_frame, int) else db.getImageCount()

    hist_mask = np.load(hist_mask_path).astype(bool)

    vecs, points, frames = read_tracks_list_by_frame(db_path, window_size=ws_angles, start_frame=min_frame, end_frame=max_frame,
                                                     return_dict=True,
                                                     track_types="all")  # reading vetctors and points as arrays

    # reading tracks and calculating angles
    vecs, points, frames, angs = angles_to_straight_line(straight_line, vecs, points, frames)
    lengths = np.linalg.norm(vecs, axis=1)  # calculating the length

    ol_vecs = np.repeat(np.expand_dims(straight_line, axis=0), len(vecs), axis=0)
    fig_angel_viz = vizualize_angles(angs, points, vecs, ol_vecs, arrows=True, image=im, normalize=True, size_factor=50,
                                     text=False, sample_factor=int(np.ceil(0.2*max_frame)), cbar_max_angle=np.pi/2)
    fig_angel_viz.savefig(os.path.join(output_folder, "anlges_vizualization_axis.png"))

    ps_in, [fs_in, as_in, ls_in] = extract_angles_area(hist_mask, points, frames, angs, lengths, dtype="array")
    ps_out, [fs_out, as_out,ls_out] = extract_angles_area(~hist_mask, points,frames, angs, lengths, dtype="array")

    # calculating the mean angle
    ma_in = get_mean_anlge_over_time(fs_in, max_frame, ws_mean, bs_mean, as_in, lengths=ls_in, weighting="nw")
    ma_out = get_mean_anlge_over_time(fs_out, max_frame, ws_mean, bs_mean, as_out, lengths=ls_out, weighting="nw" )
    fig_ao = plot_mean_angles([ma_in, ma_out],vmin=0,vmax=np.pi/2, labels=["angles close to axis", "angles distant to axis"])
    fig_ao.savefig(os.path.join(output_folder, weighting+"_"+"mean_anlge_over_time.png"))

    # writing out put csv file
    frames_out = np.arange(0, max_frame-ws_mean)[::bs_mean]
    lvs = [frames_out, ma_in ,ma_out]
    headers = ["frames","ma %s in" % weighting, "ma %s out" % weighting ]
    arr_save = np.round(np.array([np.array(x) for x in lvs if isinstance(x,(list,np.ndarray))]),5).T
    np.savetxt(out_file, np.array(arr_save), delimiter=",", fmt="%.3f", header=",".join(headers))



def angle_distance_distribution(db_path, output_folder, min_frame=0, max_frame=None, ws_angles=1, window_length=501, ymin=0, ymax=90, px_scale=None, fl=[], wl=[]):
    '''
    Calculates the distribution of the mean movement angles of cells towards the spheroid center over distance. The angle is projected to a range
    between 0° to 90°. A cell moving away and a cell moving towards the spheroid center both have an angle of 0 degrees.
    Your database needs tracks of moving cells and markers of type "center", that mark the center of spheroids. The distance angle distribution is
    created by smoothing the data with an savitzky golay filter with defined window_length. Plot and data are stored in outptfolder.
    
    :param db_path: full path to the database
    :param output_folder: output folder, is created on demand 
    :param max_frame:  maximal frame up to which tracks are analyzed. None if you want all tracks.
    :param ws_angles:  angle of cell movement is calculated using the i th and (i+ws_angles) th position (in frames)
    of the cell
    :param window_legth: Window size in data points which is used for the sliding window of the savitzky golay filter
    to create a smooth distribution - must be an odd number
    :param ymin: minimal angle to show in plot (degree)
    :param ymax: maximal angle to show in plot (degree)
    :param px_scale: Used to visualize distance in um instead pixels. Default is None and displays px values
    :return:
    '''

    #reading center positions, one image of the spheroids and identifying maximal frame if max_frame==None
    createFolder(output_folder)
    im, im_shape = None, None
    with OpenDB(db_path) as db:
        centers = np.array([(m.x, m.y) for m in db.getMarkers(type="center")])
        try:
            im = db.getImages()[0].data
            im_shape = im.shape
        except FileNotFoundError:
            print("couldn't find image and image shape for database " + db_path)
            im_shape = (2192, 2752)
        max_frame = max_frame if isinstance(max_frame, int) else db.getImageCount()

    # reading angles, movement vectors, frames and points adn randomizing cell tracks
    vecs, points, frames = read_tracks_list_by_frame(db_path, window_size=ws_angles, start_frame=min_frame, end_frame=max_frame,
                                                     return_dict=True,
                                                     track_types="all")  # reading vetctors and points as arrays
    point_dict, vec_dict, frames_dict, angs_dict, dist_vectors = angles_to_centers(centers, vecs, points, frames)
    lengths_dict = {key: np.linalg.norm(v, axis=1) for key, v in vec_dict.items()}  # calculating the length
    # randomized angles
    vecs_rot, points_rot, frames_rot = randomize_tracks(vecs, points, frames, im_shape)
    point_dict_rand, vec_dict_rand, frames_dict_rand, angs_dict_rand, dist_vectors_rand = angles_to_centers(centers,
                                                                                                            vecs_rot,
                                                                                                            points_rot,
                                                                                                            frames_rot)
    lengths_dict_rand = {key: np.linalg.norm(v, axis=1) for key, v in vec_dict_rand.items()}  # calculating the length

    # filtering and wieghting:
    FW=FilterAndWeighting(angs_dict, lengths_dict, point_dict, frames_dict)
    FW.apply_filter(fl)
    FW.apply_weighting(wl)
    angs_dict, lengths_dict, point_dict, frames_dict = FW.angles, FW.lengths, FW.points, FW.frames

    FW_rand=FilterAndWeighting(angs_dict_rand, lengths_dict_rand, point_dict_rand, frames_dict_rand)
    FW.apply_filter(fl)
    FW.apply_weighting(wl)
    angs_dict_rand, lengths_dict_rand, point_dict_rand, frames_dict_rand = FW_rand.angles, FW_rand.lengths, FW_rand.points, FW_rand.frames



     # soritng by distance and smoothing with savgol filter
    distances, angles, lengths, angle_f = analyze_angel_distacne_distribution(point_dict, angs_dict,lengths_dict, centers, window_length, output_folder, name_add="")
    distances_rand, angles_rand, lengths_rand, angle_rand_f = analyze_angel_distacne_distribution(point_dict_rand, angs_dict_rand, lengths_dict_rand, centers, window_length, output_folder, name_add="randomized_")



    fig = plot_distance_distribution(distances, angle_f, distances_rand, angle_rand_f, px_scale=px_scale, ymin=ymin,
                               ymax=ymax)

    fig.savefig(os.path.join(output_folder,"distance_distribution.png"))


    return (distances, angles)







def analyze_angel_distacne_distribution(point_dict, angs_dict, lengths_dict, centers, window_length, output_folder, name_add=""):
    # splitting into dictionary according to the closest spheroid center


    point_array = np.array(list(point_dict.values())[0])
    distances = np.linalg.norm(point_array[:, None] - centers[None, :], axis=2)
    distances = np.min(distances,axis=1) # selects only the colsest distnace (which is also how the angle is defined)
    angles = np.array(list(angs_dict.values())[0])
    lengths = np.array(list(lengths_dict.values())[0])
    # sort depending on distance
    distances, angles, lengths = zip(*sorted(zip(distances, angles, lengths)))
    distances = np.array(distances)
    angles = np.array(angles)

    # use savitzky golay filter instead of simple sliding window
    from scipy import signal
    angle_f = signal.savgol_filter(angles, window_length=window_length, polyorder=1, mode='mirror')

    # save results
    np.save(os.path.join(output_folder, name_add+'distances_px.npy'), distances)
    np.save(os.path.join(output_folder, name_add+'mean_angles_rad.npy'), angle_f)

    return (distances, angles, lengths, angle_f)

if __name__=="__main__":
    #plt.ioff()
    folder = "/home/user/Software/fiber-orientation/spheroid_spheroid_axis"
    db_path = os.path.join(folder, "db.cdb")
    hist_mask_path = os.path.join(folder, "hist_mask.npy")
    output_folder1 = os.path.join(folder, "test1")
    output_folder2 = os.path.join(folder, "test2")

    max_frame = None
    ### could use upt to 2 GB Memory and 3 minutes per database for a single analysis type

    angle_to_center_analysis(db_path, output_folder2, output_file="nw_mean_angles.txt", max_frame=max_frame,
                             ws_angles=1, ws_mean=30, bs_mean=2)
    #angle_to_center_analysis(db_path, output_folder2, output_file="lw_mean_angles.txt", max_frame=max_frame,
    #                         ws_angles=1, ws_mean=30, bs_mean=2, weighting="lw")

    #angle_to_line_analysis(db_path, hist_mask_path, output_folder1, output_file="nw_mean_angles.txt", max_frame=max_frame,
    #                       ws_angles=1, ws_mean=30, bs_mean=2, weighting="nw")
   # angle_to_line_analysis(db_path, hist_mask_path, output_folder1, output_file="lw_mean_angles.txt", max_frame=max_frame,
   #                        ws_angles=1, ws_mean=30, bs_mean=2, weighting="lw")
