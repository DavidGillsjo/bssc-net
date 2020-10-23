import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp
from scipy.linalg import null_space
from math import atan2, pi

Z_VIEW_ANGLE = 45

def getFigure(resolution=[1280, 720]):
    dpi = 200.0
    resolution = np.array(resolution, dtype=np.float)
    return plt.figure(figsize=resolution/dpi, dpi = dpi)

def compareVoxels(grid_dict, fcn, *args, **kwargs):
    dsize = len(grid_dict)
    fig = getFigure()
    for i, (name, grid) in enumerate(grid_dict.items()):
        ax = plt.subplot(1,dsize,i+1, projection='3d')
        fcn(grid, *args, **kwargs, ax=ax)
        plt.title(name)
    return fig

def plotVoxels(gridLabels, suncg_labels, vox_min, vox_unit, mask = None, save_path = None, animate = False, camera_P = None, camera_K = None, ax = None):
    if ax is None:
        fig = getFigure()
        ax = fig.gca(projection='3d')


    cmap=plt.get_cmap('tab20')
    occMask = gridLabels > 0
    if mask is not None:
        occMask &= mask
    xyz = np.nonzero(occMask)
    positions = np.vstack([xyz[0], xyz[1], xyz[2]])*vox_unit + vox_min.reshape([3,1])
    gridLabelsMasked = gridLabels[occMask]
    colors = cmap(np.mod(gridLabelsMasked, 20))

    labels = np.unique(gridLabelsMasked)
    legend_entries = []

    plot_order = [0,2,1]

    for l in labels:
        l_mask = gridLabelsMasked == l
        pos = positions[:,l_mask]
        pos_reorder = pos[plot_order]
        col = colors[l_mask]
        label_name = suncg_labels[l]


        alpha = 0.1 if label_name.lower() in ('floor', 'wall', 'ceiling', 'window') else 0.8
        legend_entries.append(label_name)
        ax.scatter(*pos_reorder, c=col, alpha=alpha, edgecolors = 'k')

    if camera_P is not None:
        camera_pos = null_space(camera_P)
        camera_pos /= camera_pos[3]
        camera_pos = camera_pos[plot_order]
        camera_front = camera_P[2,:3]
        camera_front = camera_front[plot_order]
        camera_front /= np.linalg.norm(camera_front)
        ax.quiver(*camera_pos, *camera_front, length=1.0)
        ax.view_init(Z_VIEW_ANGLE, atan2(camera_pos[1],camera_pos[0])*180/pi)

    ax.legend(legend_entries, loc='upper left')

    if save_path is None:
        return

    if animate:
        anim = FuncAnimation(fig, lambda i: ax.view_init(Z_VIEW_ANGLE, i*50), frames=np.arange(0, 6), interval=2000)
        anim.save('{}.gif'.format(save_path), dpi=80, writer='imagemagick')
    else:
        plt.savefig('{}.png'.format(save_path))
    plt.close()

    if camera_P is not None and camera_K is not None:
        img_coords = camera_K@camera_P@np.vstack([positions, np.ones(positions.shape[1])])
        img_coords /= img_coords[2]
        img_mask = (0 < img_coords[0]) & (img_coords[0] < 640) & (0 < img_coords[1]) & (img_coords[1] < 480)
        fig = getFigure()
        plt.scatter(img_coords[0,img_mask], img_coords[1,img_mask], c = colors[img_mask])
        plt.savefig('{}_proj.png'.format(save_path))
        plt.close()

def plotVoxelScalar(voxels, vox_min, vox_unit, cmap = 'gray', mask = None, save_path = None, animate = False, camera_P = None, ax = None, vmin = None, vmax = None):
    return_fig = False
    if ax is None:
        fig = getFigure()
        ax = fig.gca(projection='3d')
        return_fig = True


    if mask is None:
        mask = np.ones(score.shape, dtype=np.bool)
    xyz = np.nonzero(mask)
    plot_order = [0,2,1]
    positions = np.vstack([xyz[plot_order[0]], xyz[plot_order[1]], xyz[plot_order[2]]])*vox_unit + vox_min.reshape([3,1])
    voxels_masked = voxels[mask]

    sc = ax.scatter(*positions, c=voxels_masked, alpha=0.3, edgecolors = 'k', cmap = cmap, vmin = vmin, vmax = vmax)
    plt.colorbar(sc)

    if camera_P is not None:
        camera_pos = null_space(camera_P)
        camera_pos /= camera_pos[3]
        camera_pos = camera_pos[plot_order]
        ax.view_init(Z_VIEW_ANGLE, atan2(camera_pos[1],camera_pos[0])*180/pi)

    if save_path is not None:
        if animate:
            anim = FuncAnimation(fig, lambda i: ax.view_init(Z_VIEW_ANGLE, i*50), frames=np.arange(0, 6), interval=2000)
            anim.save('{}.gif'.format(save_path), dpi=80, writer='imagemagick')
        else:
            plt.savefig('{}.png'.format(save_path))
        plt.close()

    if return_fig:
        return fig


def plotTSDF(tsdf, save_path, vox_min, vox_unit, animate=False, flipped=False):
    tsdf_mask = np.abs(tsdf) > 0 if flipped else np.abs(tsdf) < 5*vox_unit
    xyz = np.nonzero(tsdf_mask)
    x = xyz[0]*vox_unit + vox_min[0]
    y = xyz[1]*vox_unit + vox_min[1]
    z = xyz[2]*vox_unit + vox_min[2]
    # x, y, z = np.meshgrid(*[np.arange(d) for d in tsdf.shape])


    fig = getFigure()
    ax = fig.gca(projection='3d')

    sc = ax.scatter(x, z, y, c=tsdf[tsdf_mask], alpha = 0.5)
    plt.colorbar(sc)

    if animate:
        anim = FuncAnimation(fig, lambda i: ax.view_init(30, i*50), frames=np.arange(0, 6), interval=2000)
        anim.save('{}.gif'.format(save_path), dpi=80, writer='imagemagick')
    else:
        plt.savefig('{}.png'.format(save_path))
    plt.close()

    fig = getFigure()
    plt.hist(tsdf.flat)
    plt.savefig('{}_hist.png'.format(save_path))
    plt.close()

    fig = getFigure()
    plt.hist(tsdf[tsdf_mask])
    plt.savefig('{}_hist_masked.png'.format(save_path))
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test plotting voxels')
    parser.add_argument('gt_file', type=str, help='Path to gt file')
    args = parser.parse_args()

    from ssc.data.suncg_mapping import SUNCGMapping
    import os
    import os.path as osp
    labels = SUNCGMapping()

    gt_npz = np.load(args.gt_file)
    fname = osp.splitext(osp.basename(args.gt_file))[0] + '_voxel'

    plotVoxels(gt_npz['voxels'], labels.get_classes(), gt_npz['vox_min'], gt_npz['vox_unit'], save_path = osp.join(os.getcwd(), fname))
