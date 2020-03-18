import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def vizualize_angles(angles, points, vecs1, vecs2, image=None, normalize=True, norm_factor=0.1, sample_factor=10):
    plt.figure()
    if isinstance(image, np.ndarray):
        plt.imshow(image)
    # normalization and creating a color range
    colors = matplotlib.cm.get_cmap("Greens")((angles - 0) / (np.pi / 2 - 0))
    for i,(p,v1,v2,c) in enumerate(zip(points,vecs1,vecs2,colors)):
        if i % sample_factor == 0:
            plt.arrow(p[0], p[1], v1[0], v1[1], head_width=5, color="C0")
            plt.arrow(p[0], p[1], v2[0], v2[1], head_width=5, color="C1")
            plt.scatter(p[0],p[1],color=c)
            plt.text(p[0], p[1], str(i) + "\n" + str(np.round(angles[i],2)))

    norm = matplotlib.colors.Normalize(vmin=0, vmax=np.pi / 2)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap("Greens"), norm=norm)
    plt.colorbar(sm)
