from methods.MethodTemplate import MethodTemplate
from benchmark_demo.utilstf import *
from scipy.spatial import ConvexHull, Delaunay


def counting_edges(tri,zeros):
    simplices = tri.simplices
    NZ = simplices.shape[0]
    sides = np.zeros((NZ,3))
    max_side = np.zeros((NZ,))
    for i, simplex in enumerate(simplices):
        vertex = zeros[simplex]
        sides[i,0] = np.sqrt(np.sum((vertex[0] - vertex[1]) ** 2))
        sides[i,1] = np.sqrt(np.sum((vertex[0] - vertex[2]) ** 2))
        sides[i,2] = np.sqrt(np.sum((vertex[1] - vertex[2]) ** 2))
        max_side[i] = np.max(sides[i])
    return sides, max_side


def mask_triangles(F, tri, selection):
    mask = np.zeros(F.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
           simplex = tri.find_simplex((i,j))
           if np.any(simplex == selection):
               mask[i,j] = 1

    return mask

def delaunay_triangulation_denoising(signal, params = None, return_dic = False):
    if len(signal.shape) == 1:
        signal = np.resize(signal,(1,len(signal)))

    if params is None:
        LB, UB = 2, 3
    else:
        LB, UB = params[:]

    Nfft = signal.shape[1]
    g, T = get_round_window(Nfft)
    stft, stft_padded, Npad = get_stft(signal,g)
    margin = 2

    S = np.abs(stft)**2
    zeros = find_zeros_of_spectrogram(S)
    valid_ceros = np.zeros((zeros.shape[0],),dtype=bool)
    valid_ceros[(margin<zeros[:,0]) 
                & ((S.shape[0]-margin)>zeros[:,0])
                & ((S.shape[1]-margin)>zeros[:,1])
                & (margin<zeros[:,1]) ] = True
    
    vertices = zeros/T # Normalize
    delaunay_graph = Delaunay(zeros)
    tri = delaunay_graph.simplices
    valid_tri = np.zeros((tri.shape[0],),dtype=bool)
    selection = np.zeros((tri.shape[0],),dtype=bool)
    sides, max_sides = counting_edges(delaunay_graph,vertices)

    for i,_ in enumerate(tri):
        valid_tri[i] = np.all(valid_ceros[tri[i]])
        side = max_sides[i]
        selection[i] = np.any(LB < side) & np.all(UB > side) & valid_tri[i]


    # selection = np.where((LB < max_side) & (UB > max_side))
    # selection =  & valid_tri
    tri_select = tri[selection]
    mask = mask_triangles(stft, delaunay_graph, np.where(selection))  
    signal_r, t = reconstruct_signal_2(mask, stft_padded, Npad)
    if return_dic:
        return {'s_r': signal_r,
                'mask': mask,
                'tri': tri,
                'tri_select': tri_select,
                'zeros': zeros}
    else:
        return signal_r


class NewMethod(MethodTemplate):
    def __init__(self):
        self.id = 'delaunay_triangulation'
        self.task = 'denoising'

    def method(self,signals, params = None):
        if len(signals.shape) == 1:
            signals = np.resize(signals,(1,len(signals)))
        
        signals_output = np.zeros(signals.shape)
        for i, signal in enumerate(signals):
            signals_output[i] = delaunay_triangulation_denoising(signal,params)
        return signals_output


