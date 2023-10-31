from mcsm_benchmarks.benchmark_utils import MethodTemplate
from src.utilities.utilstf import *
from scipy.spatial import Delaunay
# import matplotlib.pyplot as plt
from src.utilities.spatstats_utils import compute_scale
import matplotlib.path as mpltPath

def points_in_triangles(S,TRI,vertices):
    """_summary_

    Args:
        S (_type_): _description_
        TRI (_type_): _description_
        vertices (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    

def find_adjacent_triangles(tri, first_tri = None):
    """_summary_

    Args:
        tri (_type_): _description_
        first_tri (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
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
    """_summary_

    Args:
        S (_type_): _description_
        zeros (_type_): _description_
        tri (_type_): _description_
        ngroups (_type_, optional): _description_. Defaults to None.
        min_group_size (int, optional): _description_. Defaults to 1.
        q (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
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
            adjacent_tri, tri = find_adjacent_triangles(tri, triangle)
            if adjacent_tri is not None:
                next_triangles = np.concatenate((next_triangles,adjacent_tri),axis=0)
        
        saved_triangles = np.concatenate((saved_triangles,current_triangles),axis=0)
        
        if np.all(next_triangles == 0):
            if saved_triangles.shape[0] >= min_group_size:
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

    # ntri2 = sum(len(i) for i in groups_of_triangles) # control ntri1==ntri2

    energy_per_group = list()
    masks_of_each_group = list()
    mask = np.zeros_like(S).astype(bool)

    for group in groups_of_triangles:
        group_mask = mask_triangles3(S, group, zeros)
        mask += group_mask 
        energy_per_group.append(np.sum(S*group_mask))
        masks_of_each_group.append(group_mask) # This is memory expensive

    # Selection of groups by energy above a quantile of the total energy
    if q is not None:
        ind_group = np.where(energy_per_group > np.quantile(energy_per_group,q))
        groups_of_triangles = [groups_of_triangles[i] for i in ind_group[0]]
        masks_of_each_group = [masks_of_each_group[i] for i in ind_group[0]]
        # masks_of_each_group = [mask_triangles3(S, groups_of_triangles[i], zeros) for i in ind_group[0]]
        mask = sum(masks_of_each_group)
        mask[np.where(mask>1)] = 1
    
    # Selection of the more energetic ngroups of triangles
    if ngroups is not None:
        if ngroups == 'all' or ngroups > len(groups_of_triangles):
            ngroups = len(groups_of_triangles)
    
        order_energy_basins = np.argsort(energy_per_group)[-1:0:-1]
        groups_of_triangles = [groups_of_triangles[i] for i in order_energy_basins[0:ngroups]]
        masks_of_each_group = [masks_of_each_group[i] for i in order_energy_basins[0:ngroups]]
        # masks_of_each_group = [mask_triangles3(S, groups_of_triangles[i], zeros) for i in order_energy_basins[0:ngroups]]
        mask = sum(masks_of_each_group)
        # mask[np.where(mask>1)] = 1  

    # return the mask.
    return groups_of_triangles, mask

def describe_triangles(tri,zeros):
    """_summary_

    Args:
        tri (_type_): _description_
        zeros (_type_): _description_

    Returns:
        _type_: _description_
    """
    simplices = tri.simplices
    NZ = simplices.shape[0]
    sides = np.zeros((NZ,3))
    max_side = np.zeros((NZ,))

    area_triangle = np.zeros((NZ,))

    for i, simplex in enumerate(simplices):
        vertex = zeros[simplex]
        sides[i,0] = np.sqrt(np.sum((vertex[0] - vertex[1]) ** 2))
        sides[i,1] = np.sqrt(np.sum((vertex[0] - vertex[2]) ** 2))
        sides[i,2] = np.sqrt(np.sum((vertex[1] - vertex[2]) ** 2))
        max_side[i] = np.max(sides[i])

        area_triangle[i] = (vertex[0,0]*(vertex[1,1]-vertex[2,1])
                            + vertex[1,0]*(vertex[2,1]-vertex[0,1])
                            + vertex[2,0]*(vertex[0,1]-vertex[1,1]))


    return sides, max_side, area_triangle


# def mask_triangles(F, tri, selection):
#     mask = np.zeros(F.shape)
#     for i in range(mask.shape[0]):
#         for j in range(mask.shape[1]):
#            simplex = tri.find_simplex((i,j))
#            if np.any(simplex == selection):
#                mask[i,j] = 1

#     return mask

# def mask_triangles2(F, triangles, zeros):
#     mask = np.zeros(F.shape)
#     points = np.array([[i,j] for i in range(F.shape[0]) for j in range(F.shape[1])])
#     inside2 = np.zeros((points.shape[0],)).astype(bool)
#     for tri in triangles:
#             path = mpltPath.Path(zeros[tri,:])
#             inside2 += path.contains_points(points)   
#     for point in points[inside2,:]:         
#         mask[tuple(point)] = 1
#     return mask

def mask_triangles3(F, triangles, zeros):
    mask = np.zeros(F.shape).astype(bool)
    triangles = triangles.astype(int)
    # inside2 = np.zeros((points.shape[0],)).astype(bool)
    for tri in triangles:
            min_row, min_col = np.min(zeros[tri,:],axis=0)
            max_row, max_col = np.max(zeros[tri,:],axis=0)   
            points = np.array([[i,j] for i in range(int(max_row+1-min_row)) for j in range(int(max_col+1-min_col))])
            tri_vert = zeros[tri,:] - [int(min_row), int(min_col)]
            path = mpltPath.Path(tri_vert)
            inside2 = path.contains_points(points)
            points = points + [int(min_row), int(min_col)]
            for point in points[inside2,:]:         
                mask[tuple(point)] = True
    return mask

def compute_scale_triangles(signal, edges_signal, mc_reps=99,alpha = 0.01):
    N = len(signal)
    Nfft = 2*N
    g,T = get_round_window(Nfft)
    quantiles = np.arange(0.50,0.99,0.01)
    edge_quantiles = np.zeros((mc_reps,len(quantiles)))
    
    k = int(np.floor(alpha*(mc_reps+1)))-1 # corresponding k value
    # print(k)
    # MC simulations of noise triangles.
    for i in range(mc_reps):
        
        noise = np.random.randn(N)
        S, stft, stft_padded, Npad = get_spectrogram(noise, window = g)
        zeros = find_zeros_of_spectrogram(S)

        # Normalize the position of zeros
        vertices = zeros/T # Normalize
        delaunay_graph = Delaunay(zeros)
        # tri = delaunay_graph.simplices

        # Get the length of each edge and the areas of the triangles.
        _, longest_edges, _ = describe_triangles(delaunay_graph,vertices)
        edge_quantiles[i] = np.quantile(longest_edges, quantiles)
    
    # Compute quantiles for signal:
    edge_quantiles_empirical = np.quantile(edges_signal, quantiles)

    # Compute summary statistics:
    average_quantiles = np.mean(edge_quantiles, axis=0)
    inf_norm = lambda a: np.linalg.norm(a,np.inf)
    tm = np.zeros((mc_reps,len(quantiles)))
    texp = np.zeros((len(quantiles),))

    for q in range(1,len(quantiles)):
        for i, row in enumerate(edge_quantiles):
            tm[i,q] = inf_norm(row[0:q]-average_quantiles[0:q]) 

        texp[q] = inf_norm(edge_quantiles_empirical[0:q]-average_quantiles[0:q])

    tm = np.sort(tm, axis=0,)
    tm = tm[-1:0:-1,:]
    max_diff_t = np.argmax(texp-tm[k, :])
    scale = average_quantiles[max_diff_t]
    return scale


def delaunay_triangulation_denoising(signal,
                                    LB=1.75,
                                    UB=3.0,
                                    margin = 0,                        
                                    grouping=False, 
                                    ngroups=None, 
                                    min_group_size=1,
                                    q=None,
                                    adapt_thr=False,                            return_mask = False,   
                                    return_dic = False,
                                    Nfft = None,
                                    test_params = None,):

    """Signal filtering by domain detection using Delaunay triangulation.

    Args:
        signal (_type_): _description_
        LB (float, optional): _description_. Defaults to 1.85.
        UB (int, optional): _description_. Defaults to 3.
        return_dic (bool, optional): _description_. Defaults to False.
        grouping (bool, optional): _description_. Defaults to True.
        ngroups (_type_, optional): _description_. Defaults to None.
        min_group_size (int, optional): _description_. Defaults to 1.
        q (_type_, optional): _description_. Defaults to None.
        adapt_thr (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    # Set parameters
    N = len(signal)
    if Nfft is None:
        Nfft = 2*N

    g, T = get_round_window(Nfft)
    stft = get_stft(signal, window = g, Nfft=Nfft)

    # Computes the spectrogram and its zeros.
    S = np.abs(stft[0:Nfft//2+1,:])**2
    zeros = find_zeros_of_spectrogram(S)

    # Get only the zeros within the margins
    valid_zeros = np.zeros((zeros.shape[0],),dtype=bool)
    valid_zeros[(0<=zeros[:,0]) 
                & ((S.shape[0]-margin)>=zeros[:,0])
                & ((S.shape[1]-margin)>=zeros[:,1])
                & (0<=zeros[:,1]) ] = True
    # valid_zeros[:]=True
    # If signal is real, beware of taking zeros near the time axis
    if signal.dtype != complex128:
        valid_zeros[(T<zeros[:,0])]=True 

    # Normalize the position of zeros
    vertices = zeros/T # Normalize

    # Compute Delaunay triangulation.
    delaunay_graph = Delaunay(zeros)
    tri = delaunay_graph.simplices
    valid_tri = np.zeros((tri.shape[0],),dtype=bool)
    selection = np.zeros((tri.shape[0],),dtype=bool)

    # Get the length of each edge and the areas of the triangles.
    edges, longest_edges, area_triangle = describe_triangles(delaunay_graph,vertices)

    # Compute and adaptive threshold if its required, otherwise use "LB"
    if adapt_thr:
        if test_params is None:
            test_params = { 'alpha':0.05,
                            'fun':'Fest', 
                            'correction':'rs', 
                            'transform':'asin(sqrt(.))',
                            'rmin':0.65,
                            'rmax':1.05,                            
                        }
        
        scale_pp = compute_scale(signal,**test_params)
        LB = 2*scale_pp
        # LB = compute_scale_triangles(signal, longest_edges, mc_reps=99, alpha=0.01)
        # print('Threshold:{}'.format(LB))

    area_thr =  0 #0.3 #LB/8

    # print(area_triangle)
    # Select triangles that fulfill all the conditions
    for i,_ in enumerate(tri):
        valid_tri[i] = np.all(valid_zeros[tri[i]])
        side = longest_edges[i]
        area = area_triangle[i]
        selection[i] = (np.any(LB < side) 
                        & np.all(UB > side) 
                        & valid_tri[i]
                        & np.any(area>area_thr))

    tri_select = tri[selection]

    # Group adyacent triangles to estimate domains, and create a mask.
    if grouping:
        groups_of_triangles, mask = grouping_triangles(S, zeros, tri_select,
                                                        ngroups=ngroups,
                                                        min_group_size=min_group_size,
                                                        q = q)
    else:
        # mask = mask_triangles(stft, delaunay_graph, np.where(selection))  
        mask = mask_triangles3(S, tri_select, zeros)  

    # If just the mask is required
    if return_mask:
        return mask.astype(bool)

    # Apply the reconstruction formula to the masked STFT to filter the signal.
    # signal_r, t = reconstruct_signal_2(mask, stft_padded, Npad, Nfft)
    signal_r = np.real(reconstruct_signal_3(mask, stft, window=g))
    # Return dictionary if requested, otherwise return the denoised signal.
    if return_dic:
        return {'s_r': signal_r,
                'mask': mask.astype(bool),
                'tri': tri,
                'tri_select': tri_select,
                'zeros': zeros,
                'stft': stft}
    else:
        return signal_r

def block_based_dt_denoising(signal,block_len=1024,**kwargs):
    N = len(signal)
    t_init = 0
    mask = np.zeros((block_len+1, N),bool)
    bn=0
    signal_mean = np.mean(signal)
    # signal -= signal_mean
    while t_init<N:
        print(t_init)
        t_end = np.min([t_init+block_len,N])
        block = signal[t_init:t_end]

        if len(block) < block_len:
            aux = np.random.randn(block_len,)*0.001
            aux[0:len(block)] = block
            block = aux

        mask_block = delaunay_triangulation_denoising(block,
                                        return_mask = True, # Just get the mask.
                                        # margin= int(np.sqrt(block_len)),
                                        **kwargs)
                                        
                                        # grouping = False,
                                        # min_group_size=3,
                                        # LB = 1.7,
                                        # )
        
        mask[:,t_init:t_end] += mask_block[:,0:(t_end-t_init)]
        t_init += (block_len-block_len//4)
        bn += 1
        print('Block: {}'.format(bn))
    
    print('Invertion...')
    g, T = get_round_window(2*block_len)
    stft = get_stft(signal, window = g, Nfft=2*block_len)
    signal_r = np.real(reconstruct_signal_3(mask, stft, window=g))

    return mask, stft, signal_r#+signal_mean


class NewMethod(MethodTemplate):
    def __init__(self):
        self.id = 'delaunay_triangulation'
        self.task = 'denoising'

        # In case is needed...
        # self.cs = ComputeStatistics()

    def method(self, signal, *args, **kwargs):
        # nc = signal.total_comps
        signal_output = delaunay_triangulation_denoising(signal, *args, **kwargs)    
        return signal_output

    def get_parameters(self):            # Use it to parametrize your method.
        return [{'adapt_thr': True},]    # Use adaptive threshold.
