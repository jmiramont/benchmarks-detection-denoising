from mcsm_benchs.benchmark_utils import MethodTemplate
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
import scipy.stats as spst
import scipy.signal as sg
from src.utilities.utilstf import *
from src.utilities.spatstats_utils import compute_scale
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

def get_mask_from_centers(Sww, empty_mask, radi_expand):
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

    kdpos = KDTree(np.array([u, v]).T,compact_nodes=True)
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


def points_in_empty_ball(radius, center, tree, mascara=None):
    """_summary_

    Args:
        S (_type_): _description_
        TRI (_type_): _description_
        vertices (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # Get submasc
    minX=np.max([center[0]-radius,0])
    maxX=np.min([center[0]+radius,mascara.shape[0]])
    minY=np.max([center[1]-radius,0])
    maxY=np.min([center[1]+radius,mascara.shape[1]])

    vecx = (np.arange(minX, maxX))
    vecy = (np.arange(minY, maxY))
    g = np.transpose(np.meshgrid(vecx, vecy))

    submasc=mascara[minX:maxX,minY:maxY]
    # result = tree.query_ball_point(g, radius)

    for i in range(submasc.shape[0]):
        for j in range(submasc.shape[1]):
            punto=[minX+i,minY+j]
            result = tree.query_ball_point(punto, radius)
            submasc[i, j] = len(result) > 0


    mascara[minX:maxX,minY:maxY]=submasc
    return mascara


def get_mask_from_centers_2(Sww, empty_mask, radi_expand):
    # extract region of interest
    Nfft = Sww.shape[1]
    fmin = 0 #int(np.sqrt(Nfft))//2
    # fmax = empty_mask.shape[0] - fmin
    tmin = 0 #int(np.sqrt(Nfft))//2
    # tmax = empty_mask.shape[1] - tmin
    mask = np.zeros_like(Sww)
    # sub_empty = empty_mask[fmin:fmax, tmin:tmax]
    u, v = np.where(empty_mask)
    kdpos = KDTree(np.array([u, v]).T,compact_nodes=True)
    
    for i in range(len(u)):
        points_in_empty_ball(int(radi_expand*np.sqrt(Nfft)), [u[i], v[i]] , kdpos, mascara=mask)
    
    return mask

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
    centers = find_center_empty_balls(Sww, pos_aux, a, radi_seg=radi_seg)
    mask = get_mask_from_centers(Sww, centers, radi_expand=radi_expand)
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
