import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from utilities import normalize
import cv2
from skimage.transform import resize
from imageio import mimsave
def scatter_3D(a, cmap="jet", sca_args={}, control="color", size=60):

    # default arguments for the quiver plot. can be overwritten by quiv_args
    scatter_args = {"alpha":1}
    scatter_args.update(sca_args)

    x, y, z = np.indices(a.shape)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    fig = plt.figure()
    ax = fig.gca(projection='3d', rasterized=True)

    if control=="color":
        # make cmap
        cbound = [0, np.nanmax(a)]
        # create normalized color map for arrows
        norm = matplotlib.colors.Normalize(vmin=cbound[0], vmax=cbound[1])  # 10 ) #cbound[1] ) #)
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # different option
        colors = matplotlib.cm.jet(norm(a)).reshape(a.shape[0]*a.shape[1]*a.shape[2],4)  #
        # plotting
        ax.scatter(x, y, z, c=colors, s=size, **scatter_args)
        plt.colorbar(sm)

    if control == "alpha":
        # untested####
        col = [(0, 0, 1, x / np.max(z)) for x in np.ravel(z)]
        ax.scatter(x, y, z, c=colors, s=size, **scatter_args)
        plt.show()

    if control=="size":
        sizes = (a - a.min()) * size  / a.ptp()
        ax.scatter(x, y, z, a, s=sizes, **scatter_args)
        ax_scale = plt.axes([0.88,0.1,0.05,0.7])
        #ax_scale.set_ylim((0.1,1.2))
        nm = 5
        ax_scale.scatter([0] * nm, np.linspace(a.min(),a.max(),nm) , s=sizes.max()*np.linspace(0,1,nm))
        ax_scale.spines["left"].set_visible(False)
        ax_scale.spines["right"].set_visible(True)
        ax_scale.spines["bottom"].set_visible(False)
        ax_scale.spines["top"].set_visible(False)
        ax_scale.tick_params(axis="both", which="both", labelbottom=False,labelleft=False, labelright=True,bottom=False,left=False,right=True)


        # implement marker scale bar


    ax.set_xlim(0, a.shape[0])
    ax.set_ylim(0, a.shape[1])
    ax.set_zlim(0, a.shape[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return fig





def explode(data):
    # following "https://matplotlib.org/3.1.1/gallery/mplot3d/voxels_numpy_logo.html"

    if len(data.shape) == 3:
        size = np.array(data.shape) * 2
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
    if len(data.shape) == 4: ## color data
        size = np.array(data.shape)[:3] * 2
        data_e = np.zeros(np.concatenate([size - 1,np.array([4])]) , dtype=data.dtype)
        data_e[::2, ::2, ::2,:] = data

    return data_e


def plot_3D_alpha(data):
    # following "https://matplotlib.org/3.1.1/gallery/mplot3d/voxels_numpy_logo.html"

    col = np.zeros((data.shape[0], data.shape[1], data.shape[2], 4))


    data_fil = data.copy()
    data_fil[(data == np.inf)] = np.nanmax(data[~(data == np.inf)])
    data_fil =(data_fil - np.nanmin(data_fil)) / (np.nanmax(data_fil) - np.nanmin(data_fil))
    data_fil[np.isnan(data_fil)] = 0

    col[:, :, :, 2] = 1
    col[:, :, :, 3] = data_fil

    col_exp = explode(col)
    fill = explode(np.ones(data.shape))

    x, y, z = np.indices(np.array(fill.shape) + 1).astype(float) // 2

    x[0::2,:,:] += 0.05
    y[:,0::2,:] += 0.05
    z[:,:,0::2] += 0.05
    x[1::2,:,:] += 0.95
    y[:,1::2,:] += 0.95
    z[:,:,1::2] += 0.95


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(x,y,z,fill, facecolors=col_exp, edgecolors=col_exp)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()



def quiver_3D(u, v, w, x=None, y=None, z=None, image_dim=None, mask_filtered=None, filter_def=0, filter_reg=[1], cmap="jet", quiv_args={}, cbound=None):
    #filter_def filters values with smaler absolute deformation
    # nans are also removed
    # setting the filter to <0 will probably mess up the arrow colors
    # filter_reg filters every n-th value, separate for x, y, z axis
    # you can also provide your own mask with mask_filtered !!! make sure to filter out arrows with zero total deformation!!!!
    # other wise the arrows are not colored correctly
    # use indices for x,y,z axis as default - can be specified by x,y,z

    # default arguments for the quiver plot. can be overwritten by quiv_args
    quiver_args = {"normalize":False, "alpha":0.8, "pivot":'tail', "linewidth":1, "length":20}
    quiver_args.update(quiv_args)

    u = np.array(u)
    v = np.array(v)
    w = np.array(w)

    if not isinstance(image_dim, (list, tuple, np.ndarray)):
        image_dim = np.array(u.shape)

    # generating coordinates if not provided
    if x is None:
        # if you provide deformations as a list
        if len(u.shape) == 1:
            x, y, z = [np.indices(u.shape)[0]  for i in range(3)]
        # if you provide deformations as an array
        elif len(u.shape) == 3:
            x, y, z = np.indices(u.shape)
        else:
            raise ValueError("displacement data has wrong number of dimensions (%s). Use 1d array or list, or 3d array."%str(len(u.shape)))

    # multiplying coordinates with "image_dim" factor if coordinates are provided
    x, y, z = np.array([x,y,z]) #* np.expand_dims(np.array(image_dim) / np.array(u.shape),axis=list(range(1,len(u.shape)+1)))

    deformation = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    if not isinstance(mask_filtered, np.ndarray):
        mask_filtered = deformation > filter_def
        if isinstance(filter_reg, list):
            show_only = np.zeros(u.shape).astype(bool)
            if len(filter_reg) == 1:
                show_only[::filter_reg[0]] = True
            elif len(filter_reg) == 3:
                show_only[::filter_reg[0], ::filter_reg[1], ::filter_reg[2]] = True
            else:
                raise ValueError(
                    "filter_reg data has wrong length (%s). Use list with length 1 or 3." % str(len(filter_reg.shape)))
            mask_filtered = np.logical_and(mask_filtered, show_only)



    xf = x[mask_filtered]
    yf = y[mask_filtered]
    zf = z[mask_filtered]
    uf = u[mask_filtered]
    vf = v[mask_filtered]
    wf = w[mask_filtered]
    df = deformation[mask_filtered]

    # make cmap
    if not cbound:
        cbound = [0,np.nanmax(df)]
    # create normalized color map for arrows
    norm = matplotlib.colors.Normalize(vmin=cbound[0], vmax=cbound[1])  # 10 ) #cbound[1] ) #)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # different option
    colors = matplotlib.cm.jet(norm(df))  #

    colors = [c for c, d in zip(colors, df) if d > 0] + list(chain(*[[c, c] for c, d in zip(colors, df) if d > 0]))
    # colors in ax.quiver 3d is really fucked up/ will probably change with updates:
    # requires list with: first len(u) entries define the colors of the shaft, then the next len(u)*2 entries define
    # the color ofleft and right arrow head side in alternating order. Try for example:
    # colors = ["red" for i in range(len(cf))] + list(chain(*[["blue", "yellow"] for i in range(len(cf))]))
    # to see this effect
    # BUT WAIT THERS MORE: zeor length arrows are apparently filtered out in the matplolib with out filtering the color list appropriately
    # so we have to do this our selfs as well

    # plotting
    fig = plt.figure()
    ax = fig.gca(projection='3d', rasterized=True)

    ax.quiver(xf, yf, zf, vf, uf, wf, colors=colors, **quiver_args)
    plt.colorbar(sm)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.w_xaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
    ax.w_yaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
    ax.w_zaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
    return fig
# plotting


def animate_stacks(stack1, stack2, interval=100, repeat_delay=0, z_range=[0,"max"], drift_correction=True,
                   normalize_images=True, add_circle=False, pos=(300,300), save_gif=True, gif_name="out.gif",max_axis=2,im_aspect="auto"):
    # gif of maximum projections of stack1 and stack2

    if z_range[1] == "max":
        z_range[1] = stack1.shape[2]
    ims = [np.max(stack1[:, :, z_range[0]: z_range[1]], axis=max_axis), np.max(stack2[:, :, z_range[0]: z_range[1]], axis=max_axis)]


    # representation of the image stacks by maximums projections. The red circle marks the position of the cell
    def update_plot(i, ims, ax):
        a1 = ax.imshow(ims[i],aspect=im_aspect)

        if add_circle:
            a2 = ax.add_patch(plt.Circle(pos, 100, color="red", fill=False))
        else:
            a2=None
        if ax.texts:
            ax.texts[0].remove()
        a3 = ax.text(30, 30, "stack " + str(i),fontsize=25)
        return [a1, a2, a3]


    bg_1 = np.percentile(ims[1], 25)
    if drift_correction:
        from scipy.ndimage.interpolation import shift
        from skimage.feature import register_translation
        shift_values = register_translation(ims[0], ims[1], upsample_factor=10)
        shift_y = shift_values[0][0]
        shift_x = shift_values[0][1]
        print("shift between images = ", shift_values[0])
        ims[1] = shift(ims[1], shift=(shift_y, shift_x), order=3,cval=bg_1)

    if normalize_images:
       ims[0] = normalize(ims[0],0.1,99.1)
       ims[1] = normalize(ims[1],0.1,99.1)

    if save_gif:
        max_shape = np.max([im.shape for im in ims])
        ims_out = []
        for i, im in enumerate(ims):
            im =  resize(im,(max_shape,max_shape) )
            im = cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            font = cv2.FONT_HERSHEY_SIMPLEX
            im = cv2.putText(im, "frame " + str(i), (50, 50), font, 2, (255, 255, 255), 4)
            ims_out.append(im)
        mimsave(gif_name, ims_out, duration=1)


    fig = plt.figure()
    ax = plt.gca()
    ani = animation.FuncAnimation(fig, update_plot, 2, interval=interval, blit=repeat_delay, repeat_delay=0,
                                  fargs=(ims, ax))
    return ani, ims

