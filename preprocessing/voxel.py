import _init_binvox
import time
from binvox_rw import Voxels
import numpy as np
from scipy.linalg import block_diag

from matplotlib import path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from matplotlib.patches import Polygon, Rectangle

from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint
from shapely.prepared import prep
from shapely.ops import polygonize
import os.path as osp
from utils import getFigure
import math
import pickle
import binvox_rw
from sklearn.neighbors import NearestNeighbors

class NNCache:
    def __init__(self, cache_dir, model_dir):
        self.cache_dir = cache_dir
        self.model_dir = model_dir

    def query(self, X, obj_id, radius):
        picke_fname = osp.join(self.cache_dir, '{}.pickle'.format(obj_id))
        try:
            with open(picke_fname, 'rb') as f:
                nbrs = pickle.load(f)
                mask = self._get_NN_mask(nbrs, X, radius)
        except FileNotFoundError:
            binvox_filename = osp.join(self.model_dir, obj_id, '{}.binvox'.format(obj_id))
            with open(binvox_filename, 'rb') as f:
                obj_bv = binvox_rw.read_as_coord_array(f, fix_coords=False)
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(obj_bv.data.T)

            with open(picke_fname, 'wb') as f:
                pickle.dump(nbrs, f)
            mask = self._get_NN_mask(nbrs, X, radius)

        return mask

    def _get_NN_mask(self, nbrs, X, radius):
        distances, indices = nbrs.kneighbors(X.T)
        matched = distances.ravel() < radius
        return matched


class VoxelSearch:
    def __init__(self, voxels, radius = 1.1):
        assert radius > 1
        self.occupied = voxels
        cube_side = math.ceil(2*radius)
        center = np.full(3, cube_side/2.0)
        mask = np.zeros(3*[cube_side], dtype=np.bool)
        for i, m in enumerate(mask.flat):
            multi_index = np.unravel_index(i, mask.shape)
            dist = np.linalg.norm(multi_index - center + 0.5)
            mask[multi_index] = (dist <= radius)
        self.directions = np.array(tuple(np.nonzero(mask))).T

    # Greedy NN
    def _single_nn(self, center):
        self.center = center
        self.queue = [np.floor(center).astype(np.int)]
        self.visited_and_queued = set(tuple(self.queue[0]))

        while len(self.queue) > 0:
            vox_idx = self.queue.pop()

            #Within radius?
            # print('dist', (vox_idx + 0.5 - self.center), np.sqrt(np.sum((vox_idx + 0.5 - self.center)**2)))
            if np.sum((vox_idx + 0.5 - self.center)**2) > self.radius2:
                continue

            #Only use valid index
            if np.any(vox_idx < 0) or np.any(vox_idx >= self.occupied.shape[0]):
                continue

            #Is this our neighbour?
            if self.occupied[tuple(vox_idx)]:
                return True

            # Add neighbouring voxels for exploration
            new_idx = self.directions + vox_idx
            for idx in new_idx:
                idx_tuple = tuple(idx)
                if not idx_tuple in self.visited_and_queued:
                    self.visited_and_queued.add(idx_tuple)
                    self.queue.append(idx)

        #Nothing found
        return False

    def _single_nn_simple(self, center):
        center_idx = np.round(center).astype(np.int)
        query_idx = self.directions + center_idx
        valid_mask = np.all(query_idx > 0, axis=1)  & np.all(query_idx < self.occupied.shape[0], axis=1)
        return np.any(self.occupied[tuple(query_idx[valid_mask].T)])


    def nn_search(self, coords, radius = None):
        assert coords.shape[1] == 3, "coords must be Nx3"
        if radius:
            self.radius2 = radius**2
            return np.array([self._single_nn(c) for c in coords], dtype=np.bool)
        else:
            return np.array([self._single_nn_simple(c) for c in coords], dtype=np.bool)





class SUNCGCamera:
    def __init__(self, csvrow):
        self.pos = np.array(csvrow[:3], dtype=np.float)
        self.front = np.array(csvrow[4:7], dtype=np.float)
        self.up = np.array(csvrow[8:11], dtype=np.float)
        self.P = camera2P(self.pos, self.up, self.front)

def constructK():
    cam_vertical_fov = 60 #Degrees
    im_height = 480
    im_width = 600
    focal_l = im_height/(2 * np.tan((cam_vertical_fov/2) * np.pi/180))
    K = np.zeros([3,3])
    K[0,0] = K[1,1] = focal_l
    K[0,2] = im_width/2
    K[1,2] = im_height/2
    K[2,2] = 1
    return K

def boxOverlap(box1_min, box1_max, box2_min, box2_max):
    overlap = True
    for i in range(box1_min.size):
        overlap &= (box1_min[i] < box2_max[i] and
                    box2_min[i] < box1_max[i])
    return overlap

def sameFloor(voxOriginWorld, bbox_min, bbox_max):
    return (bbox_min[1] < voxOriginWorld[1] and
            voxOriginWorld[1] < bbox_max[1] )

def _plotPoly(points, poly, s_poly, debug_dir):
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(poly[:,0], poly[:,1])
    plt.plot(points[:,0], points[:,1], '.')
    plt.axis('equal')
    plt.subplot(2,1,2)
    plt.plot(points[:,0], points[:,1], '.')

    try:
        poly_iter = iter(s_poly)
    except TypeError:
        poly_iter = [s_poly]

    for clean_poly in poly_iter:
        try:
            poly_coords = np.array(list(clean_poly.exterior.coords))
        except AttributeError:
            continue
        try:
            poly_coords2 = [np.array(list(c.coords)) for c in clean_poly.interiors]
        except TypeError:
            poly_coords2 = [np.array(list(clean_poly.interiors.coords))]
        plt.plot(poly_coords[:,0], poly_coords[:,1])
        for c in poly_coords2:
            plt.plot(c[:,0], c[:,1],'--')
    plt.axis('equal')
    import uuid
    plt.savefig(osp.join(debug_dir, 'Multi{}.png'.format(uuid.uuid4().hex)))
    plt.close()

def inPolygon(points, poly, debug_dir = None, convex_hull = False):
    assert points.shape[1] == 2, "Only handles 2D!"

    #Check that polygon is valid
    if np.any(np.isnan(poly)):
        return np.zeros(points.shape[0], dtype=np.bool)

    if convex_hull:
        s_poly = MultiPoint(poly).convex_hull
    else:
        s_poly = Polygon(poly)

    if not s_poly.is_valid:
        # Buffer is used to trim overlapping segments
        s_poly = s_poly.buffer(0)

    if debug_dir:
        _plotPoly(points, poly, s_poly, debug_dir)

    # Buffer will split into multiPoly if necessary
    try:
        poly_iterator = iter(s_poly)
    except TypeError:
        return _inCleanPolygon(points, s_poly)

    inside = np.zeros(points.shape[0], dtype=np.bool)

    for clean_poly in poly_iterator:
        inside |= _inCleanPolygon(points, clean_poly)

    return inside


def _inCleanPolygon(points, s_poly):
    if s_poly.exterior is None:
        return np.zeros(points.shape[0], dtype=np.bool)

    bounds = s_poly.bounds
    inside = ((bounds[0] < points[:,0]) & (points[:,0] < bounds[2]) &
              (bounds[1] < points[:,1]) & (points[:,1] < bounds[3]))

    # The actual polygon check, Change to Shapely if not precise enough.
    # Though shapely takes a lot more time for computation
    poly_path = path.Path(list(s_poly.exterior.coords))
    inside[inside] = poly_path.contains_points(points[inside])

    return inside

def camera2P(pos, up, front):
    right = np.cross(front, up)
    right /= np.linalg.norm(right)

    #Solve for rotation matrix
    A = np.vstack([
        block_diag(right, right, right),
        block_diag(up, up, up),
        block_diag(front, front, front)
    ])

    b = np.array([
        1, 0, 0,  #right ->  x
        0, -1, 0, #up    -> -y
        0, 0, 1   #front ->  z
    ]).reshape([9,1])

    R = np.linalg.solve(A,b)
    R = R.reshape(3,3)
    t = -R.dot(pos)

    assert np.abs(1-np.linalg.det(R)) < 1e-3
    assert np.linalg.norm(R[2,:3] - front) < 1e-3
    assert np.linalg.norm(-R.T.dot(t) - pos) < 1e-3

    return np.hstack([R, t.reshape([3,1])])


# From stackoverflow
def _cuboid_data(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def _plotCubeAt(positions,sizes=None,colors=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( _cuboid_data(p, size=s) )
    return Poly3DCollection(np.concatenate(g),
                            facecolors=np.repeat(colors,6, axis=0), **kwargs)


def plotVoxelsList(gridLabels, gridCoords, suncg_labels, save_path = None):

    fig = getFigure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    cmap=plt.get_cmap('tab20')
    occMask = gridLabels > 0

    gridLabelsMasked = gridLabels[occMask]
    gridCoordsMasked = gridCoords[:,occMask]

    colors = cmap(np.mod(gridLabelsMasked, 20))

    labels = np.unique(gridLabelsMasked)
    legend_entries = []

    for l in labels:
        l_mask = gridLabelsMasked == l
        pos = gridCoordsMasked[:,l_mask]
        col = colors[l_mask]
        label_name = suncg_labels.getClassRootFromRootID(l)


        alpha = 0.1 if label_name.lower() in ('floor', 'wall', 'ceiling', 'window') else 0.8
        legend_entries.append(label_name)
        ax.scatter(pos[0], pos[2], pos[1], c=col, alpha=alpha, edgecolors = 'k')

    plt.legend(legend_entries, loc = 'upper left')
    anim = FuncAnimation(fig, lambda i: ax.view_init(30, i*50), frames=np.arange(0, 6), interval=2000)
    anim.save(save_path, dpi=fig.get_dpi(), writer='imagemagick')
    plt.close()
