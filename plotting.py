import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def vizualize_angles(angles, points, vecs1, vecs2, image=None, normalize=True, size_factor=10, sample_factor=10,cbar_max_angle=None):
    plt.figure()
    if isinstance(image, np.ndarray):
        plt.imshow(image)
    # normalization and creating a color range
    colors = matplotlib.cm.get_cmap("Greens")(angles/np.max(angles))
    for i,(p,v1,v2,c) in enumerate(zip(points,vecs1,vecs2,colors)):
        if i % sample_factor == 0:
            if normalize:
                v1 = v1 * size_factor / np.linalg.norm(v1)
                v2 = v2 * size_factor / np.linalg.norm(v2)
            plt.arrow(p[0], p[1], v1[0], v1[1], head_width=5, color="C0")
            plt.arrow(p[0], p[1], v2[0], v2[1], head_width=5, color="C1")
            plt.scatter(p[0],p[1],color=c)
            plt.text(p[0], p[1], str(i) + "\n" + str(np.round(angles[i],2)))
    vmax = cbar_max_angle if isinstance(cbar_max_angle,(int,float)) else np.max(angles)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap("Greens"), norm=norm)
    plt.colorbar(sm)
