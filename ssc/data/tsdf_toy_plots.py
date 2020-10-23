import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    d = np.linspace(-2,2,100)
    f = {}
    tsdf = np.ones_like(d)*np.sign(d)
    trunc_mask  = np.abs(d) < 1
    tsdf[trunc_mask] = d[trunc_mask]
    f['tsdf'] = tsdf

    f['sign(d)(1-abs(d))'] = np.sign(tsdf)*(1 - np.abs(tsdf))
    f['sign(d)*1-d'] = np.sign(tsdf)*1 - tsdf
    # f['sign(d)*1-d)'] = np.sign(tsdf)*1 - tsdf

    plt.figure()
    for key, item in f.items():
        plt.plot(d, item, label=key)
    plt.legend()
    plt.savefig('tsdf_toy_plot.png')
    plt.close()
