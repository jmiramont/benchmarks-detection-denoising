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

def delaunay_triangulation_denoising(signal,params):
    if len(signal.shape) == 1:
        signal = np.resize(signal,(1,len(signal)))

    signals_output = np.zeros(signal.shape)
    Nfft = signal.shape[1]
    g, T = get_round_window(Nfft)
    stft, stft_padded, Npad = get_stft(signal,g)
    S = np.abs(stft)**2
    pos = find_zeros_of_spectrogram(S)
    vertices = pos/T # Normalize
    tri = Delaunay(pos)
    sides, max_side = counting_edges(tri,vertices)
    LB = 2
    UB = 3
    selection = np.where((LB < max_side) & (UB > max_side))
    tri_select = tri.simplices[selection]
    mask = mask_triangles(stft, tri, selection)  
    signal_r, t = reconstruct_signal_2(mask, stft_padded, Npad)
    return signal_r, mask


class NewMethod(MethodTemplate):
    def method(self,signals,params):
        signals_output = np.zeros(signals.shape)
        for i, signal in enumerate(signals):
            signals_output[i],_ = delaunay_triangulation_denoising(signal,params)
        return signals_output

def return_method_instance():
    return NewMethod('denoising','delaunay_triangulation')


