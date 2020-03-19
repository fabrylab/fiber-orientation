
####################### functions that where used to make a2d histogramm of angles ##########################
def fill_nans_nearest_neighbour(arr,nan_size=np.inf):
    #zeropadding 1 pixel at each edge
    arr_zero_padded = np.zeros((arr.shape[0] + 2, arr.shape[1] + 2))+np.nan
    arr_zero_padded[1:-1, 1:-1] = arr

    nan_mask = np.isnan(arr_zero_padded).astype(bool)
    add_upper = copy.deepcopy(arr_zero_padded)
    add_lower = copy.deepcopy(arr_zero_padded)
    add_right = copy.deepcopy(arr_zero_padded)
    add_left = copy.deepcopy(arr_zero_padded)
    # fillling in shifted values
    add_upper[0:-2, 1:-1]=arr_zero_padded[1:-1, 1:-1]
    add_lower[2:, 1:-1]=arr_zero_padded[1:-1, 1:-1]
    add_right[1:-1, 0:-2]=arr_zero_padded[1:-1, 1:-1]
    add_left[0:-2, 2:]=arr_zero_padded[1:-1, 1:-1]
    #removing all non nan values in the original array
    add_upper[~nan_mask] = np.nan
    add_lower[~nan_mask] = np.nan
    add_right[~nan_mask] = np.nan
    add_left[~nan_mask] = np.nan

    # mean for points that had multiple non-nan neighbours
    add_upper_c = np.where(~np.isnan(add_upper))
    add_lower_c = np.where(~np.isnan(add_lower))
    add_right_c = np.where(~np.isnan(add_right))
    add_left_c = np.where(~np.isnan(add_left))
    all_coordinates = np.concatenate(
        [np.array(add_upper_c), np.array(add_lower_c), np.array(add_right_c), np.array(add_left_c)], axis=1)
    all_values = np.concatenate(
        [add_upper[add_upper_c], add_lower[add_lower_c], add_right[add_right_c], add_left[add_left_c]])
    coords_dict = defaultdict(list)
    for c1, c2, v in zip(all_coordinates[0], all_coordinates[1], all_values):
        coords_dict[(c1, c2)].append(v)
    coords_dict = {key: np.mean(value) for key, value in coords_dict.items()}

    # adding the coordinates to the original array
    for (c1, c2), value in coords_dict.items():
        arr_zero_padded[c1, c2] = value
    arr_ret = arr_zero_padded[1:-1, 1:-1]  # retrieve original shape

    # filling all areas that where nan in the original array with the mean of values that have been added in these areas
    nan_mask_og = np.isnan(arr)
    objects = label(nan_mask_og)
    regions = regionprops(objects)
    for r in regions:
        if r.area>nan_size: # only filling holes up to a certain size
            coords = r.coords
            arr_ret[coords[:, 0], coords[:, 1]] = np.nanmean(arr_ret[coords[:, 0], coords[:, 1]])

    return arr_ret,nan_mask_og

def filter_mask_for_size(mask,size):
    mask_cp = copy.deepcopy(mask)
    objects = label(mask)
    regions = regionprops(objects)
    for r in regions:
        if r.area<size: # only filling holes up to a certain size
            coords = r.coords
            mask_cp[coords[:,0],coords[:,1]]=0
    return mask_cp


def grid_by_mean(positions,values,shape,window_size=20,fill_nans=False,nan_size=np.inf,fill_nan_smooth=False,raw=False):

    new_dims=(int(np.ceil(shape[0]/window_size))+1,int(np.ceil(shape[1]/window_size))+1)
    array=np.zeros(new_dims)
    number=np.zeros(new_dims)
    pos=np.round(positions/window_size).astype(int)

    for p,v in tqdm(zip(pos,values)):
        array[p[0],p[1]]+=v
        number[p[0],p[1]]+=1
    number[number==0]=np.nan
    mean_array=array/number

    mean_array,nan_mask=fill_nans_nearest_neighbour(mean_array,nan_size=0)

    if raw:
        mean_rs = cv2.resize(mean_array, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        nan_rs = cv2.resize(nan_mask.astype(float), (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        mean_rs[nan_rs.astype(bool)]=np.nan
        return mean_rs
    out=zoom(mean_array,(window_size,window_size)) # restoring original shape
    nan_mask=zoom(np.isnan(number),(window_size,window_size),order=0) # order zero is nearest neighbour?


    if fill_nans:
        nan_mask_f = filter_mask_for_size(nan_mask, nan_size)
        if fill_nan_smooth:
            nan_mask_f=gaussian_filter(nan_mask_f.astype(float),sigma=70)
            nan_mask_f=nan_mask_f>0.1
        out[nan_mask_f] = np.nan

    else:  # refilling nans if desired
        out[nan_mask] = np.nan
    return out


def grid_with_scipy(positions,values,shape,filter_max=None,sigma=None):
    grid_x, grid_y = np.mgrid[0:shape[0] + 1:1, 0:shape[1] + 1:1]
    g_x = griddata(positions, values, (grid_x, grid_y), method='cubic')
    if isinstance(filter_max,(int,float)):
        g_x[g_x>filter_max]=filter_max
    if isinstance(sigma, (int, float)):
        g_x=gaussian_filter(g_x,sigma=sigma)
    return g_x


def get_smooth_angle_distribution(tracks, method="mean bin",window_size=30,sigma=20,nan_size=4000,fill_nan_smooth=False,fill_nans=False, raw=False):
    # calculating angles
    angles = get_angles(tracks, straight_line)
    angles_adjusted = np.abs(angles.flatten())
    angles_adjusted[angles_adjusted > (np.pi / 2)] = angles_adjusted[angles_adjusted > (np.pi / 2)] - (np.pi / 2)
    mask = np.isnan(angles_adjusted)
    angles_adjusted = angles_adjusted[~mask]
    # retriving angle positions

    positions_x = tracks[:, :-1, 0].flatten()
    positions_y = tracks[:, :-1, 1].flatten()
    positions_x = positions_x[~mask]
    positions_y = positions_y[~mask]
    positions = np.concatenate([np.expand_dims(positions_y, axis=1), np.expand_dims(positions_x, axis=1)], axis=1)

    if method == "mean bin":
        g = grid_by_mean(positions, angles_adjusted, shape, window_size=window_size, fill_nans=fill_nans, nan_size=nan_size,fill_nan_smooth=fill_nan_smooth, raw=raw)
    if method == "mean convolve":
        g = grid_by_mean_filter(positions, angles_adjusted, shape, window_size=window_size, fill_nans=fill_nans,
                         nan_size=nan_size, fill_nan_smooth=fill_nan_smooth)
    if method == "scipy griddata":
        g = grid_with_scipy(positions, angles_adjusted, shape, filter_max=np.pi / 2, sigma=sigma)
    return g,angles_adjusted,positions

def grid_by_mean_filter(positions,values,shape,window_size=20,fill_nans=False,nan_size=np.inf,fill_nan_smooth=False):

    new_dims=(int(np.ceil(shape[0]))+1,int(np.ceil(shape[1]))+1)
    array=np.zeros(new_dims)
    number=np.zeros(new_dims)
    pos=np.round(positions).astype(int)

    for p,v in tqdm(zip(pos,values)):
        array[p[0],p[1]]+=v
        number[p[0],p[1]]+=1

    number[number == 0] = 1
    mean_array = array / number
    non_zero=(~(mean_array==0)).astype(int)
    sigma=window_size

    conv1=convolve2d(mean_array,np.ones((window_size,window_size))) # sum of all none_zero elements
    conv2=convolve2d(non_zero, np.ones((window_size,window_size))) # number of non zero elements in windos
    conv2[conv2==0]=np.nan
    out=conv1/conv2 # calculating the mean for all windows
    out, nan_mask = fill_nans_nearest_neighbour(out, nan_size=0)
    out=gaussian_filter(out,sigma=200) #additional gaussian

    if fill_nans:
        nan_mask_f = filter_mask_for_size(nan_mask, nan_size)
        if fill_nan_smooth:
            nan_mask_f=gaussian_filter(nan_mask_f.astype(float),sigma=70)
            nan_mask_f=nan_mask_f>0.1
        out[nan_mask_f] = np.nan

    else:  # refilling nans if desired
        out[nan_mask] = np.nan
    return out
