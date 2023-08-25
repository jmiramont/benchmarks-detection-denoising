from benchmark_tools.benchmark_utils import MethodTemplate
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
import scipy.stats as spst
import scipy.signal as sg
from benchmark_tools.utilstf import *
from benchmark_tools.spatstats_utils import compute_scale
# from numba import njit


def find_center_empty_balls(Sww, pos_exp, a, radi_seg):
    # define a kd-tree with zeros
    kdpos = KDTree(pos_exp)

    # define a grid corresponding to the time-frequency paving
    vecx = (np.arange(0, Sww.shape[0])/a)
    vecy = (np.arange(0, Sww.shape[1])/a)
    g = np.transpose(np.meshgrid(vecy, vecx))
    result = kdpos.query_ball_point(g, radi_seg).T

    empty_mask = np.zeros(result.shape, dtype=bool)
    for i in range(len(vecx)):
        for j in range(len(vecy)):
            empty_mask[i,j] = len(result[i, j]) < 1

    return empty_mask

# def paint_empty_balls(Sww, pos_exp, a, radi_seg):
#     radi_seg = int(np.floor(radi_seg * a))
#     # define a kd-tree with zeros
#     kdpos = KDTree((pos_exp*a).astype(int))

#     # define a grid corresponding to the time-frequency paving
#     vecx = (np.arange(0, Sww.shape[0]))
#     vecy = (np.arange(0, Sww.shape[1]))
#     g = np.transpose(np.meshgrid(vecy, vecx))
#     g.resize(Sww.size,2)
#     # Find the TF points without zeros around in a radius of T.
#     result = kdpos.query_ball_point(g, radi_seg).T

#     # Create a circular patch to move:
#     gridTree = KDTree(g)
#     ref_point = np.array((Sww.shape[1]//2, Sww.shape[0]//2))
#     idx = gridTree.query_ball_point(ref_point, radi_seg)
#     patch_idx = np.floor(g[idx]).astype(int) - ref_point

#     # Paint balls of radius T around those points.
#     mask = np.zeros_like(Sww)
#     for point in g[np.where([result[i] == [] for i in range(len(result))])]:
#         aux_patch = patch_idx + point
#         aux_patch = np.delete(aux_patch, np.where(np.logical_or(aux_patch[:,0]<0, aux_patch[:,0]>=Sww.shape[1])), axis=0)
#         aux_patch = np.delete(aux_patch, np.where(np.logical_or(aux_patch[:,1]<0, aux_patch[:,1]>=Sww.shape[0])), axis=0)          
#         for idx in aux_patch:
#             mask[idx[1],idx[0]] = True


#     return mask    


def get_convex_hull(Sww, pos_exp, empty_mask, radi_expand):
    # extract region of interest
    Nfft = Sww.shape[1]
    fmin = 0 #int(np.sqrt(Nfft))//2
    fmax = empty_mask.shape[0] - fmin
    tmin = 0 #int(np.sqrt(Nfft))//2
    tmax = empty_mask.shape[1] - tmin

    sub_empty = empty_mask[fmin:fmax, tmin:tmax]
    vecx = (np.arange(0, sub_empty.shape[0]))
    vecy = (np.arange(0, sub_empty.shape[1]))
    g = np.transpose(np.meshgrid(vecx, vecy))
    u, v = np.where(sub_empty)
    kdpos = KDTree(np.array([u, v]).T)
    # result = kdpos.query_ball_point(g, radi_expand*np.sqrt(Nfft))
    sub_empty = np.zeros(sub_empty.shape, dtype=bool)
    for i in range(sub_empty.shape[1]):
        for j in range(sub_empty.shape[0]):
            result = kdpos.query_ball_point(g[j,i], radi_expand*np.sqrt(Nfft))
            sub_empty[j, i] = len(result) > 0

    u, v = np.where(sub_empty)
    # points = np.array([u, v]).T
    # hull_d = Delaunay(points) # for convenience
    mask = np.zeros(Sww.shape)
    mask[fmin:fmax, tmin:tmax] = sub_empty
    return mask #,hull_d, 


def empty_space_denoising(signal,
                            radi_seg=0.9,
                            radi_expand=None,
                            adapt_thr=False,
                            test_params = None,
                            return_dic=False,
                            ):

    radi_expand = radi_seg
    N = len(signal)
    Nfft = 2*N

        # Compute and adaptive threshold if its required, otherwise use "LB"
    if adapt_thr:
        if test_params is None:
            test_params = {
                            'fun':'Fest', 
                            'correction':'rs', 
                            'transform':'asin(sqrt(.))',
                            'rmin':0.65,
                            'rmax':1.05,                            
                        }
        
        scale_pp = compute_scale(signal,**test_params)
        radi_seg = scale_pp # Override LB
        radi_expand = scale_pp

    
    g, a = get_round_window(Nfft)
    stft = get_stft(signal, window = g, Nfft=Nfft)
    # Computes the spectrogram and its zeros.
    Sww = np.abs(stft[0:Nfft//2+1,:])**2

    pos = find_zeros_of_spectrogram(Sww)
    pos_aux = pos.copy()
    pos_aux[:,0] = pos[:,1]/a
    pos_aux[:,1] = pos[:,0]/a
    empty_mask = find_center_empty_balls(Sww, pos_aux, a, radi_seg=radi_seg)
    mask = get_convex_hull(Sww, pos_aux, empty_mask, radi_expand=radi_expand)
    # mask = paint_empty_balls(Sww, pos_aux, a, radi_seg=radi_seg)
    xr = np.real(reconstruct_signal_3(mask, stft, window=g))

    if return_dic:
        return {'s_r': xr,
                'mask': mask,
                'zeros':pos}
    else:
        return xr


class NewMethod(MethodTemplate):
    def __init__(self):
        self.id = 'empty_space'
        self.task = 'denoising'

    def method(self, signal, *args, **kwargs):
        signal_output = empty_space_denoising(signal, *args, **kwargs)    
        return signal_output
    
    def get_parameters(self):            # Use it to parametrize your method.
        return [{'adapt_thr': True},]    # Use adaptive threshold.
