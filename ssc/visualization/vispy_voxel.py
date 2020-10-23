import vispy
vispy.use(app='egl')

from moviepy.editor import VideoClip
import numpy as np
from vispy import scene, io, visuals
from vispy.color import *
import cv2

# Check the application correctly picked up egl
assert vispy.app.use_app().backend_name == 'egl', 'Not using EGL'

class AlphaAwareCM(BaseColormap):
    def __init__(self, color_list):
        bins = np.linspace(0,1,len(color_list)+1)
        self.glsl_map = 'vec4 translucent_grays(float t) {\n'
        for c_idx, (i1, i2) in enumerate(zip(bins[:-1], bins[1:])):
            return_vec = 'return vec4({0[0]:.4},{0[1]:.4},{0[2]:.4},{0[3]:.4});'.format(color_list[c_idx].rgba.flat)
            if c_idx == 0:
                self.glsl_map += '  if (t < {:.2}) {{\n    {}\n  }}'.format(i2, return_vec)
            elif c_idx == len(color_list):
                self.glsl_map += '  else {{}\n    {}\n  }}'.format(return_vec)
            else:
                self.glsl_map += '  else if (({:.2} <= t) && (t < {:.2})) {{\n    {}\n  }}'.format(i1, i2, return_vec)
        self.glsl_map += '\n}'
        super().__init__()

def plot_voxels(gridLabels, suncg_labels, vox_min, vox_unit, save_path = None, animate = False):
    nbr_classes = len(suncg_labels)

    canvas = scene.SceneCanvas(keys='interactive', bgcolor='w', size = (1920,1080))
    view = canvas.central_widget.add_view()
    azimuth = 30
    view.camera = scene.TurntableCamera(up='y', distance=4, fov=70,
                    azimuth=azimuth, elevation=30.)

    # Sample colormap and adjust alpha
    colormap = get_colormap('cubehelix')
    cm_sampled = []
    for i, (iclass, sample_f) in enumerate(zip(suncg_labels, np.linspace(0,1,nbr_classes))):
        if iclass.lower() in ('free', 'ceiling'):
            alpha = 0
        elif iclass.lower() in ('floor', 'wall', 'window'):
            alpha = 0.6
        else:
            alpha = 1.0
        cm_sampled.append(Color(color=colormap[sample_f].rgb, alpha=alpha))
    my_cm = AlphaAwareCM(cm_sampled)

    volume = scene.visuals.Volume(gridLabels, relative_step_size = 0.1, method='mip', parent=view.scene, cmap = my_cm, clim = [0, nbr_classes-1], emulate_texture = False)
    volume.transform = scene.transforms.MatrixTransform()
    volume.transform.scale(3*[vox_unit])
    volume.transform.translate(3*[-vox_unit*gridLabels.shape[0]/2.0])

    if save_path is None:
        return

    def make_frame(t):
        view.camera.set_state({'azimuth': azimuth+t*90})
        return canvas.render()

    if animate:
        animation = VideoClip(make_frame, duration=3)
        animation.write_gif('voxel.gif', fps=8, opt='OptimizePlus')
    else:
        img = canvas.render()
        cv2.imwrite('voxel.png', img[::-1])

def scatter_plot_voxels(gridLabels, suncg_labels, vox_min, vox_unit, save_path = None, animate = False):
    nbr_classes = len(suncg_labels)

    occMask = gridLabels > 0
    xyz = np.nonzero(occMask)
    positions = np.vstack([xyz[0], xyz[1], xyz[2]])
    gridLabelsMasked = gridLabels[occMask]

    canvas = scene.SceneCanvas(keys='interactive', bgcolor='w', size = (1920,1080))
    view = canvas.central_widget.add_view()
    azimuth = 30
    view.camera = scene.TurntableCamera(up='y', distance=4, fov=70,
                    azimuth=azimuth, elevation=30.)

    # Sample colormap and adjust alpha
    colormap = get_colormap('hsl', value=1.0, saturation=0.8, ncolors = nbr_classes)
    pos_color = np.zeros((positions.shape[1], 4))
    cm_sampled = []
    for i, (iclass, sample_f) in enumerate(zip(suncg_labels[1:], np.linspace(0,1,nbr_classes-1))):
        if iclass.lower() in ('floor', 'wall', 'window'):
            alpha = 0.5
        elif iclass.lower() == 'ceiling':
            alpha = 0.0
        else:
            alpha = 1.0
        base_color = colormap[sample_f].rgba.flatten()
        base_color[3] = alpha
        pos_color[i==gridLabelsMasked] = base_color

    Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
    p1 = Scatter3D(parent=view.scene)
    p1.set_gl_state('translucent', blend=True, depth_test=True)
    p1.set_data(positions.T, face_color=pos_color, symbol='disc', size=10,
            edge_width=0.5, edge_color='k')

    p1.transform = scene.transforms.MatrixTransform()
    p1.transform.scale(3*[vox_unit])
    p1.transform.translate(3*[-vox_unit*gridLabels.shape[0]/2.0])

    if save_path is None:
        return

    def make_frame(t):
        view.camera.set_state({'azimuth': azimuth+t*90})
        return canvas.render()

    if animate:
        animation = VideoClip(make_frame, duration=3)
        animation.write_gif('voxel.gif', fps=8, opt='OptimizePlus')
    else:
        img = canvas.render()
        cv2.imwrite('voxel.png', img[::-1])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test plotting voxels')
    parser.add_argument('gt_file', type=str, help='Path to gt file')
    parser.add_argument('--animate', action='store_true', help='Yield GIF instead of PNG')
    args = parser.parse_args()

    from ssc.data.suncg_mapping import SUNCGMapping
    import os
    labels = SUNCGMapping()

    gt_npz = np.load(args.gt_file)

    scatter_plot_voxels(gt_npz['voxels'], labels.get_classes(), gt_npz['vox_min'], gt_npz['vox_unit'], save_path = os.getcwd() , animate = args.animate)
