from methods.MethodTemplate import MethodTemplate
from benchmark_demo.utilstf import *
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt


def pointsInTriangles(S,TRI,vertices):
    mascara=np.zeros_like(S)
    #Coordenadas de los vertices:
    # vertTRI = np.zeros((3,2))
    vertTRI = vertices[TRI.astype(int),:]

    AT  = ( vertTRI[0,0]*(vertTRI[1,1]-vertTRI[2,1]) + 
            vertTRI[1,0]*(vertTRI[2,1]-vertTRI[0,1]) + 
            vertTRI[2,0]*(vertTRI[0,1]-vertTRI[1,1]))

    minX=int(np.min(vertTRI[:,0]))
    maxX=int(np.max(vertTRI[:,0]))
    minY=int(np.min(vertTRI[:,1]))
    maxY=int(np.max(vertTRI[:,1]))

    submasc=mascara[minX:maxX,minY:maxY]

    for i in range(submasc.shape[0]):
        for j in range(submasc.shape[1]):
            punto=[minX+i,minY+j]
            
            A1=vertTRI[0,0]*(vertTRI[1,1]-punto[1])+vertTRI[1,0]*(punto[1]-vertTRI[0,1])+punto[0]*(vertTRI[0,1]-vertTRI[1,1])
            
            A2=vertTRI[0,0]*(punto[1]-vertTRI[2,1])+punto[0]*(vertTRI[2,1]-vertTRI[0,1])+vertTRI[2,0]*(vertTRI[0,1]-punto[1])
            
            A3 = punto[0]*(vertTRI[1,1]-vertTRI[2,1])+vertTRI[1,0]*(vertTRI[2,1]-punto[1])+vertTRI[2,0]*(punto[1]-vertTRI[1,1])
            
            if np.sum(np.abs([A1, A2, A3]))==AT:
                submasc[i,j]=1
        
    mascara[minX:maxX,minY:maxY]=submasc
    return mascara
    

def adjacent_triangle(tri, first_tri = None):
    # Find adjacent triangles of first_tri.
    Nselect=tri.shape[0]
    ladosCompartidos = np.zeros((Nselect,3))
    for i in range(3):
        for j in range(3):
            ladosCompartidos[:,i] = ladosCompartidos[:,i] + (tri[:,i]==first_tri[j])

    triangulos_adyacentes= np.sum(ladosCompartidos,axis=1) >= 2

    first_tri = np.resize(first_tri, (1,3))
    # If no adjacent triangles are found, just return first_tri.
    if sum(triangulos_adyacentes)==0:
        return None, tri
    else:
    # Otherwise, save adjacent triangles, and look for their own adjacent.        
        this_adjacent = tri[triangulos_adyacentes,:]
        tri = np.delete(tri, triangulos_adyacentes, axis = 0)
        return this_adjacent, tri
    
def grouping_triangles(S, zeros, tri, ngroups=None, min_group_size=1, q = None):
    groups_of_triangles = list()
    ntri = tri.shape[0]

    if q is None and ngroups is None:
        ngroups = 'all'

    if q is not None:
        ngroups = None

    if ngroups is not None:
        q = None

    current_triangles=tri[0]
    current_triangles = np.resize(current_triangles,(1,3))
    tri = tri[1::]
    saved_triangles = np.zeros((1,3))
    while tri.size > 0:
        next_triangles = np.zeros((1,3))
        for triangle in current_triangles:
            adjacent_tri, tri = adjacent_triangle(tri, triangle)
            if adjacent_tri is not None:
                next_triangles = np.concatenate((next_triangles,adjacent_tri),axis=0)
        
        saved_triangles = np.concatenate((saved_triangles,current_triangles),axis=0)
        
        if np.all(next_triangles == 0):
            if saved_triangles.shape[0] >= min_group_size+1:
                groups_of_triangles.append(saved_triangles[1::])
            
            saved_triangles = np.zeros((1,3))
            current_triangles = tri
            current_triangles=tri[0]
            current_triangles = np.resize(current_triangles,(1,3))
            tri = tri[1::]
            if tri.size == 0:
                saved_triangles = np.concatenate((saved_triangles,current_triangles),axis=0)
                groups_of_triangles.append(saved_triangles[1::])

        else:
            current_triangles = next_triangles[1::]
        
        

    ntri2 = sum(len(i) for i in groups_of_triangles) # control ntri1==ntri2

    energy_per_group = list()
    masks_of_each_group = list()

    for group in groups_of_triangles:
        mask = np.zeros_like(S)
        for triangle in group:
            mask = mask + pointsInTriangles(S, triangle, zeros)

        energy_per_group.append(np.sum(S*mask))
        masks_of_each_group.append(mask)

    if q is not None:
        ind_group = np.where(energy_per_group > np.quantile(energy_per_group,q))
        groups_of_triangles = [groups_of_triangles[i] for i in ind_group[0]]
        masks_of_each_group = [masks_of_each_group[i] for i in ind_group[0]]

    if ngroups is not None:
        if ngroups == 'all' or ngroups > len(groups_of_triangles):
            ngroups = len(groups_of_triangles)
    
        order_energy_basins = np.argsort(energy_per_group)[-1:0:-1]
        groups_of_triangles = [groups_of_triangles[i] for i in order_energy_basins[0:ngroups]]
        masks_of_each_group = [masks_of_each_group[i] for i in order_energy_basins[0:ngroups]]
    
    mask = sum(masks_of_each_group)
    mask[np.where(mask>1)] = 1  

    return groups_of_triangles, mask

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

def delaunay_triangulation_denoising(signal,
                                    LB=1.85,
                                    UB=3, 
                                    return_dic = False,
                                    grouping=True, 
                                    ngroups=None, 
                                    min_group_size=1,
                                    q = None):
    if len(signal.shape) == 1:
        signal = np.resize(signal,(1,len(signal)))

    Nfft = 2*signal.shape[1]
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
    if grouping:
        groups_of_triangles, mask = grouping_triangles(S, zeros, tri_select,
                                                        ngroups=ngroups,
                                                        min_group_size=min_group_size,
                                                        q = q)
    else:
        mask = mask_triangles(stft, delaunay_graph, np.where(selection))  

    # mask[:] = 1
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

    def method(self,signals, params):
        if len(signals.shape) == 1:
            signals = np.resize(signals,(1,len(signals)))
        
        signals_output = np.zeros(signals.shape)
        for i, signal in enumerate(signals):
            if params is None:
                signals_output[i] = delaunay_triangulation_denoising(signal)
            else:
                signals_output[i] = delaunay_triangulation_denoising(signal, **params)    

        return signals_output



