from cell_movement_orientation import *

output_folder = "/home/user/Desktop/biophysDS/abauer/cell_orientation_22_04_2020"
createFolder(output_folder)
input_folder="/home/user/Desktop/biophysDS/abauer/tracks - marked center"
dbs=[x for x in os.listdir(input_folder) if x.endswith(".cdb")]



db = "/home/user/Desktop/biophysDS/abauer/tracks - marked center/20190304-150631_Mic3__pos003db.cdb"

db_path = os.path.join(input_folder, db)
output_folder2 = os.path.join(output_folder,db[:-4])
max_frame = None

angle_to_center_analysis(db_path, output_folder2, output_file="nw_mean_angles.txt", max_frame=max_frame,
                         ws_angles=1, ws_mean=30, bs_mean=2, weighting="nw", mark_center=True)

angle_distance_distribution(db_path, output_folder2, max_frame=max_frame, ws_angles=1, window_length=int(300/(4.095/10)+1), ymin=0, ymax=90,
                             px_scale=4.0954/10)




for db in dbs:
    print(db)
    db_path = os.path.join(input_folder, db)
    output_folder2 = os.path.join(output_folder,db[:-4])
    max_frame = None

    angle_to_center_analysis(db_path, output_folder2, output_file="nw_mean_angles.txt", max_frame=max_frame,
                             ws_angles=1, ws_mean=30, bs_mean=2, weighting="nw", mark_center=True)

    angle_distance_distribution(db_path, output_folder2, max_frame=max_frame, ws_angles=1, window_length=int(300/(4.095/10)+1), ymin=0, ymax=90,
                                px_scale=4.0954/10)
