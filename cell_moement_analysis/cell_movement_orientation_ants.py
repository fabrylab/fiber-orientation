from cell_moement_analysis.cell_movement_orientation import *
from cell_moement_analysis.angel_calculations import FilterAndWeighting
import clickpoints

db_path = "/home/user/Desktop/biophysDS/abauer/ants_out/not_stitcheddatabase.cdb"
db=clickpoints.DataFile(db_path,"r")

'''
for i in range(0,18):
    output_folder = "/home/user/Desktop/biophysDS/abauer/ants_out/analysis_frame_" + str(i)
    createFolder(output_folder)

    # frame window
    min_frame = i*10000
    max_frame = (i+1)*10000

    angle_to_center_analysis(db, output_folder, output_file="nw_mean_angles.txt", min_frame=min_frame,
                             max_frame=max_frame,
                             ws_angles=1, ws_mean=30, bs_mean=2, weighting="nw", mark_center=True)

    angle_distance_distribution(db, output_folder, min_frame=min_frame, max_frame=max_frame, ws_angles=1,
                                window_length=int(300 / (4.095 / 10) + 1), ymin=0, ymax=90,
                                px_scale=4.0954 / 10)


'''

filter_list=[(FilterAndWeighting.length_threshold,{"threshold":7}),(FilterAndWeighting.spatial_filter_radius,{"center":(332,736), "radius":70})]
weighting_list=[(FilterAndWeighting.linear_weigthing,{})]
output_folder = "/home/user/Desktop/biophysDS/abauer/ants_out/analysis_filters"
createFolder(output_folder)

# frame window
min_frame = 10000
max_frame = 15000

angle_to_center_analysis(db, output_folder, output_file="nw_mean_angles.txt", min_frame=min_frame,
                         max_frame=max_frame,
                         ws_angles=1, ws_mean=30, bs_mean=2, mark_center=True, fl = filter_list, wl = weighting_list)
angle_distance_distribution(db, output_folder, min_frame=min_frame, max_frame=max_frame, ws_angles=1,
                            window_length=int(300 / (4.095 / 10) + 1), ymin=0, ymax=90,
                            px_scale=4.0954 / 10, fl = filter_list, wl = weighting_list)

output_folder = "/home/user/Desktop/biophysDS/abauer/ants_out/analysis_no_filters"
angle_to_center_analysis(db, output_folder, output_file="nw_mean_angles.txt", min_frame=min_frame,
                         max_frame=max_frame,
                         ws_angles=1, ws_mean=30, bs_mean=2, mark_center=True)




