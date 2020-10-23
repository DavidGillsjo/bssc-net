import numpy as np
import cv2
from mayavi import mlab
mlab.options.offscreen = True
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from math import atan2, pi
import seaborn as sns
FIG_SIZE = (480, 360)
PLOT_ORDER = [0,2,1]


def compare_voxels(grid_dict, *args, **kwargs):
    dsize = len(grid_dict)
    img = np.zeros((2*FIG_SIZE[1], dsize*2*FIG_SIZE[0], 3), dtype=np.uint8)
    for i, (name, grid) in enumerate(grid_dict.items()):
        img[:, i*2*FIG_SIZE[0]:(i+1)*2*FIG_SIZE[0]] = plot_voxels(grid, *args, title=name, **kwargs)
    return img

def plot_voxels(gridValues, vox_min, vox_unit, alpha = 1.0,
                suncg_labels = None, title = 'Title', mask = None, save_path = None,
                camera_P = None, cmap = 'jet', scalar = False, vmax = None, vmin = None,
                crossection = False):
    fig = mlab.figure(size=FIG_SIZE, bgcolor=(1, 1, 1))

    #VTK cannot handle numpy bool
    if gridValues.dtype == np.bool:
        gridValues = gridValues.astype(np.uint8)

    my_mask = np.ones(gridValues.shape, dtype=np.bool)
    if not scalar:
        my_mask &= gridValues > 0

    if mask is not None:
        my_mask &= mask
    elif suncg_labels and 'ceiling' in suncg_labels:
        my_mask &= (gridValues != suncg_labels.index('ceiling'))

    # Early abort
    if not np.any(my_mask):
        mlab.close(fig)
        return np.zeros((2*FIG_SIZE[1], 2*FIG_SIZE[0], 3), dtype=np.uint8)



    xyz = np.nonzero(my_mask)
    positions = np.vstack([xyz[0], xyz[1], xyz[2]])*vox_unit + vox_min.reshape([3,1])
    gridValuesMasked = gridValues[my_mask]

    if crossection:
        cs_mask = np.zeros_like(my_mask)
        cs_mask[my_mask.shape[0]//2] = 1
        cs_mask[:, 3] = 1
        cs_mask[:, :, my_mask.shape[0]//2] = 1
        non_cs_mask = (~cs_mask)[my_mask]
        cs_mask = cs_mask[my_mask]
        positions_opaque = positions[:,non_cs_mask]
        gridValuesMasked_opaque = gridValuesMasked[non_cs_mask]

        positions = positions[:,cs_mask]
        gridValuesMasked = gridValuesMasked[cs_mask]

        cs_alpha = 0.05

    if scalar:
        mlab.points3d(*positions[PLOT_ORDER], gridValuesMasked, mode="cube", colormap=cmap, scale_factor=0.07, scale_mode='none', vmax=vmax, vmin=vmin, opacity = alpha)
        if crossection:
            mlab.points3d(*positions_opaque[PLOT_ORDER], gridValuesMasked_opaque, mode="cube", colormap=cmap, scale_factor=0.07, scale_mode='none', vmax=vmax, vmin=vmin, opacity = cs_alpha)
        lut_manager = mlab.colorbar(orientation='vertical')
        lut_manager.label_text_property.color = (0,0,0)
        lut_manager.title_text_property.color = (0,0,0)
    else:
        nbr_classes = len(suncg_labels)-1
        if nbr_classes == 1:
            mlab.points3d(*positions[PLOT_ORDER], mode="cube", color=(0.5,0.5,0.5), scale_factor=0.07, scale_mode='none', opacity = alpha)
            if crossection:
                mlab.points3d(*positions_opaque[PLOT_ORDER], mode="cube", color=(0.5,0.5,0.5), scale_factor=0.07, scale_mode='none', opacity = cs_alpha)
        else:
            pplot = mlab.points3d(*positions[PLOT_ORDER], gridValuesMasked, mode="cube", colormap='jet', scale_factor=0.07, scale_mode='none', vmax=nbr_classes, vmin=1, opacity = alpha)
            if crossection:
                pplot = mlab.points3d(*positions_opaque[PLOT_ORDER], gridValuesMasked_opaque, mode="cube", colormap='jet', scale_factor=0.07, scale_mode='none', vmax=nbr_classes, vmin=1, opacity = cs_alpha)
            #Set custom colormap
            cmap = sns.hls_palette(nbr_classes)
            pplot.module_manager.scalar_lut_manager.lut.table = 255*np.array([(*rgb, 1) for rgb in cmap])


    if camera_P is not None:
        camera_pos = null_space(camera_P)
        camera_pos /= camera_pos[3]
        camera_pos = camera_pos[PLOT_ORDER]
        camera_front = camera_P[2,:3]
        camera_front = camera_front[PLOT_ORDER]
        camera_front /= np.linalg.norm(camera_front)
        mlab.quiver3d(*camera_pos, *camera_front)
        mlab.view(azimuth = atan2(camera_pos[1],camera_pos[0])*180/pi)

    azimuth, _, dist, _ = mlab.view()
    img = np.zeros((2*FIG_SIZE[1], 2*FIG_SIZE[0], 3), dtype=np.uint8)
    for r in range(2):
        for c in range(2):
            azimuth += (2*r + c)*90
            mlab.view(azimuth = azimuth, distance = dist*0.9)
            img[r*FIG_SIZE[1]:(r+1)*FIG_SIZE[1],
                c*FIG_SIZE[0]:(c+1)*FIG_SIZE[0]] = mlab.screenshot(figure=fig, mode='rgb', antialiased=False)
            if scalar and (c+r)==0:
                lut_manager.show_legend = False

    mlab.clf(fig)
    mlab.close(fig)

    if title:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2
        text_size, _ = cv2.getTextSize(title, font, fontScale, thickness)
        text_pos = ((img.shape[1] - text_size[0])//2,
                    (img.shape[0] + text_size[1])//2)
        cv2.putText(img, title, text_pos, font, fontScale, (0,0,0), thickness=thickness)

    if save_path is not None:
        cv2.imwrite('{}.png'.format(save_path), img[:,:,::-1])

    return img


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test plotting voxels')
    parser.add_argument('gt_files', type=str, help='List of Paths to gt files', nargs='+')
    args = parser.parse_args()

    from ssc.data.suncg_mapping import SUNCGMapping
    import os
    import os.path as osp
    labels = SUNCGMapping()

    for f in args.gt_files:
        gt_npz = np.load(f)
        fname = osp.splitext(osp.basename(f))[0]

        plot_voxels(gt_npz['voxels'], gt_npz['vox_min'], gt_npz['vox_unit'], suncg_labels=labels.get_classes(), save_path = osp.join(os.getcwd(), fname), scalar=False)
