# migration angles towards spheroid-spheroid angles
# produces images and videos of spatial angle distribution
# and histogram of angles in a selected area

#from pyTFM import utilities_TFM
import sys
import re
import matplotlib.pyplot as plt
sys.path.append("/home/user/Software/fiber-orientation")
from plotting import plot_binned_angle_fileds, vizualize_angles, plot_mean_angles
from database_functions import *
from angles import *
from contextlib import suppress
from utilities import *


folder = "/home/user/Software/fiber-orientation/spheroid_spheroid_axis"
db_path = os.path.join(folder, "db.cdb")
hist_mask = os.path.join(folder,"hist_mask.npy")
out_path = os.path.join(folder, "out")


def read_tracks_randomized(db, window_size=1, start_frame=0, end_frame=None):

    with OpenDB(db) as db_l:
        if not isinstance(end_frame,int):
            end_frame = db_l.getImageCount()
        tracks = db_l.getTracksNanPadded(start_frame=start_frame, end_frame=end_frame)
         # calcualting vectors

        vecs_ret = {}
        points_ret = {}
        frames_ret = {}
        for i, v in enumerate(tracks):
            ps_rot = rotate_track(v, np.random.uniform(0,np.pi*2))
            vecs = ps_rot[window_size:]-ps_rot[:-window_size]
            nan_filter = ~np.isnan(vecs[:, 0])
            vecs_ret[i] = vecs[nan_filter]
            points_ret[i] = ps_rot[:-window_size][nan_filter]
            frames_ret[i] = np.arange(start_frame, end_frame-window_size, dtype=int)[nan_filter]
        remove_empty_entries([vecs_ret, points_ret, frames_ret], dtype="list")
    return  vecs_ret, points_ret, frames_ret



def angles_to_straight_line(db, straight_line, window_size=1, max_frame=None):

    vecs, points, frames = read_tracks_list_by_frame(db, window_size=window_size, end_frame=max_frame,  return_dict=False)
    angs = calculate_angle(vecs, straight_line, axis=1)
    angs = project_angle(angs)
    return vecs, points, frames, angs



def angles_to_centers(db, centers, window_size=1, max_frame=None):

    vecs, points, frames = read_tracks_list_by_frame(db, window_size=window_size, end_frame=max_frame, return_dict=False)
    # associate to point to closest center
    distances = np.linalg.norm(points[:,None] - centers[None,:], axis=2) #axis0->points,axis1->centers
    mins = np.argmin(distances,axis=1)
    n = range(len(centers))
    point_dict = {i:points[mins==i] for i in n}
    vec_dict = {i:vecs[mins==i] for i in n}
    frames_dict = {i:frames[mins==i] for i in n}
    dist_vectors = {i: point_dict[i] - centers[i] for i in n}

    angs_dict = {i:project_angle(calculate_angle(vec_dict[i], dist_vectors[i], axis=1)) for i in range(len(centers))}

    return point_dict, vec_dict, frames_dict, angs_dict, dist_vectors




def angles_to_centers_randomized(db, centers, window_size=1, max_frame=None):

    vecs, points, frames =  read_tracks_randomized(db, window_size=window_size, start_frame=0, end_frame=max_frame)

    points, vecs, frames = flatten_dict(points, vecs, frames)

    # associate to point to closest center
    distances = np.linalg.norm(points[:,None] - centers[None,:], axis=2) #axis0->points,axis1->centers
    mins = np.argmin(distances,axis=1)
    n = range(len(centers))
    point_dict = {i:points[mins==i] for i in n}
    vec_dict = {i:vecs[mins==i] for i in n}
    frames_dict = {i:frames[mins==i] for i in n}
    dist_vectors = {i: point_dict[i] - centers[i] for i in n}

    angs_dict = {i:project_angle(calculate_angle(vec_dict[i], dist_vectors[i], axis=1)) for i in range(len(centers))}

    return point_dict, vec_dict, frames_dict, angs_dict, dist_vectors






def read_tracks_to_binned_dict(db, binsize_time, step_size, max_frame=None):
    straight_line = get_orientation_line(db_path)
    vecs, points, frames, angs = angles_to_straight_line(db, straight_line, max_frame=max_frame)
    max_frame = np.max(frames) if not isinstance(max_frame, (int, float)) else max_frame
    # might be to memory intensive like this
    vecs_b, points_b, angs_b = binning_by_frame(frames, max_frame, binsize_time, step_size, vecs, points, angs)

    return vecs_b, points_b, angs_b




def make_spheroid_spheroid_orientation_vids(db_path,hist_mask,out_path):
    straight_line = get_orientation_line(db_path)
    remove_single_img = True

    db = clickpoints.DataFile(db_path, "r")
    im = db.getImage().data

    max_frame = db.getImageCount()  # that would be all frames

    vecs, points, frames = read_tracks_list_by_frame(db, end_frame=max_frame)
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



def angle_to_center_analysis(db_path, output_folder, output_file, max_frame=None, ws_angles=1, ws_mean=30, bs_mean=2, weighting="nw"):
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
    im = None
    createFolder(output_folder)
    out_file = os.path.join(output_folder, output_file)
    with OpenDB(db_path) as db:
        centers = np.array([(m.x, m.y) for m in db.getMarkers(type="center")])
        with suppress(FileNotFoundError): im = db.getImages()[0].data
        max_frame = max_frame if isinstance(max_frame, int) else db.getImageCount()

    # reading angles, movement vectors, frames and points adn randomizing cell tracks
    point_dict, vec_dict, frames_dict, angs_dict, dist_vectors = angles_to_centers(db_path, centers,
                                                                    window_size=ws_angles, max_frame=max_frame)
    lengths_dict = {key:np.linalg.norm(v, axis=1) for key,v in vec_dict.items()}   # calculating the length
    point_dict_rand, vec_dict_rand, frames_dict_rand, angs_dict_rand, dist_vectors_rand = angles_to_centers_randomized(db_path,
                                                centers, window_size=ws_angles, max_frame=max_frame)
    lengths_dict_rand = {key: np.linalg.norm(v, axis=1) for key, v in vec_dict_rand.items()}  # calculating the length
    point, vec, frames, angs, dists = flatten_dict(point_dict, vec_dict, frames_dict, angs_dict, dist_vectors)
    point_rand, vec_rand, frames_rand, angs_rand, dists_rand = flatten_dict(point_dict_rand, vec_dict_rand, frames_dict_rand,
                                                                            angs_dict_rand, dist_vectors_rand)
    # vizualizing angles
    fig_angel_viz=vizualize_angles(angs, point, vec, dists, arrows=True, image=im, normalize=True, size_factor=50,
                     text=False, sample_factor=int(np.ceil(0.2*max_frame)))
    fig_angel_viz.savefig(os.path.join(output_folder, "anlges_vizualization.png"))

    fig_angel_viz_rand = vizualize_angles(angs_rand, point_rand, vec_rand, dists_rand, arrows=True, image=im, normalize=True, size_factor=50,
                                     text=False, sample_factor=int(np.ceil(0.2*max_frame)))
    fig_angel_viz_rand.savefig(os.path.join(output_folder, "anlges_vizualization_randomized.png"))

    # calculating mean angles over time
    mas = []
    ma_rands = []
    for key in frames_dict.keys():
        mas.append(get_mean_anlge_over_time(frames_dict[key], max_frame, ws_mean, bs_mean,
                                            angs_dict[key], lengths=lengths_dict[key], weighting=weighting))
        ma_rands.append(get_mean_anlge_over_time(frames_dict_rand[key], max_frame, ws_mean,
                                                 bs_mean, angs_dict_rand[key], lengths=lengths_dict_rand[key], weighting=weighting))

    # plotting mean angles over time
    labels=["spheroid " + str(i+1) for i in range(len(mas))] + ["randomly reorientated"] * len(ma_rands) #
    fig_unweighted = plot_mean_angles(mas+ma_rands,vmin=0,vmax=np.pi/2,labels=labels)
    fig_unweighted.savefig(os.path.join(output_folder,weighting+"_"+"mean_anlge_over_time.png"))

    # saving mean and randomized angles
    frames_out = np.arange(0, max_frame-ws_mean)[::bs_mean]
    lvs = [frames_out]+ mas+ ma_rands
    headers = ["frames"] + ["angle to sph %s" % str(i) for i in range(len(mas))] + \
              ["angle to sph %s randomized" % str(i) for i in range(len(mas))]
    arr_save = np.round(np.array([np.array(x) for x in lvs if isinstance(x,(list,np.ndarray))]),5).T
    np.savetxt(out_file, np.array(arr_save), delimiter=",", fmt="%.3f", header=",".join(headers))



def angle_to_line_analysis(db_path, hist_mask_path, output_folder, output_file="mean_anlges.txt", max_frame=None, ws_angles=1, ws_mean=30, bs_mean=2,
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

    frames_out, ma_in_nw, ma_out_nw, ma_in_lw, ma_out_lw, im = None, None, None, None, None, None
    createFolder(output_folder)
    out_file = os.path.join(output_folder,output_file)
    with OpenDB(db_path) as db:
        with suppress(FileNotFoundError): im = db.getImages()[0].data
        e_points = [(m.x, m.y) for m in db.getMarkers(type="straight_line")]
        straight_line = np.array(e_points[0]) - np.array(e_points[1])
        max_frame = max_frame if isinstance(max_frame, int) else db.getImageCount()

    hist_mask = np.load(hist_mask_path).astype(bool)
    # reading tracks and calculating angles
    vecs, points, frames, angs = angles_to_straight_line(db_path, straight_line, window_size=ws_angles, max_frame=max_frame)
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



def angle_distance_distribution(db_path, output_folder, max_frame=None,  ws_angles=1, window_length=501, ymin=0,ymax=90, px_scale = None):
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
    im = None
    createFolder(output_folder)
    with OpenDB(db_path) as db:
        centers = np.array([(m.x, m.y) for m in db.getMarkers(type="center")])
        with suppress(FileNotFoundError): im = db.getImages()[0].data
        max_frame = max_frame if isinstance(max_frame, int) else db.getImageCount()

    # reading angles, movement vectors, frames and points adn randomizing cell tracks
    point_dict, vec_dict, frames_dict, angs_dict, dist_vectors = angles_to_centers(db_path, centers,
                                                                    window_size=ws_angles, max_frame=max_frame)
    point_array =  np.array(list(point_dict.values())[0])
    distances = np.linalg.norm(point_array[:,None] - centers[None,:], axis=2) 
    angles =  np.array(list(angs_dict.values())[0])

    # sort depending on distance
    distances, angles = zip(*sorted( zip( distances, angles)  ))
    distances = np.array(distances)
    angles = np.array(angles)
    
    # use savitzky golay filter instead of simple sliding window
    from scipy import signal
    angle_f = signal.savgol_filter(angles, window_length=window_length, polyorder=1,   mode='mirror' )

    # save results
    np.save(os.path.join(output_folder, 'distances_px.npy'), distances)
    np.save(os.path.join(output_folder, 'mean_angles_rad.npy'), angle_f)
    
    
    # plot results
    plt.figure()
    plt.grid('True')
    if px_scale is None:
        plt.plot(distances,angle_f*(360/(2*np.pi)),'--', c='orange', label='Tracks')
        plt.plot(distances,[45]*len(distances),'--', c='k', label='Random')
        plt.xlabel('Distance')
    else:
        plt.plot(distances*px_scale,angle_f*(360/(2*np.pi)),'--', c='orange', label='Tracks')
        plt.plot(distances*px_scale, [45]*len(distances),'--', c='k', label='Random')
        plt.xlabel('Distance (µm)')
    plt.ylabel('Mean Angle to spheroid (°)')
    plt.ylim(ymin,ymax)
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'angle_distance.png'), dpi=350 )
    plt.close()
 
    return (distances, angles)



if __name__=="__main__":
    #plt.ioff()
    folder = "/home/user/Software/fiber-orientation/spheroid_spheroid_axis"
    db_path = os.path.join(folder, "db.cdb")
    hist_mask_path = os.path.join(folder, "hist_mask.npy")
    output_folder1 = os.path.join(folder, "out1")
    output_folder2 = os.path.join(folder, "out2")

    max_frame=200
    ### could use upt to 2 GB Memory and 3 minutes per database for a single analysis type

    angle_to_center_analysis(db_path, output_folder2, output_file="nw_mean_angles.txt", max_frame=max_frame,
                             ws_angles=1, ws_mean=30, bs_mean=2, weighting="nw")
    angle_to_center_analysis(db_path, output_folder2, output_file="lw_mean_angles.txt", max_frame=max_frame,
                             ws_angles=1, ws_mean=30, bs_mean=2, weighting="lw")

    angle_to_line_analysis(db_path, hist_mask_path, output_folder1, output_file="nw_mean_angles.txt", max_frame=max_frame,
                           ws_angles=1, ws_mean=30, bs_mean=2, weighting="nw")
    angle_to_line_analysis(db_path, hist_mask_path, output_folder1, output_file="lw_mean_angles.txt", max_frame=max_frame,
                           ws_angles=1, ws_mean=30, bs_mean=2, weighting="lw")
